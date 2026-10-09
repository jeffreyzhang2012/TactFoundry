"""Flow matching with explicit instruction-only, pure-noise training cases."""
import dataclasses
import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models import pi0, pi0_config, model as model_module


def noisy_actions(rng, actions, probability):
    noise_key,time_key,choice_key=jax.random.split(rng,3)
    noise=jax.random.normal(noise_key,actions.shape)
    times=jax.random.beta(time_key,1.5,1.,actions.shape[:-2])*.999+.001
    pure=jax.random.bernoulli(choice_key,probability,actions.shape[:-2])
    # Share noise for instruction-only examples, removing a nuisance variable
    # when matched observations with different instructions enter a batch.
    noise=jnp.where(pure[...,None,None],noise[:1],noise)
    times=jnp.where(pure,1.,times)
    mixed=times[...,None,None]*noise+(1.-times[...,None,None])*actions
    return mixed,noise-actions,times


class ConditionalPi0(pi0.Pi0):
    def __init__(self,config,rngs):
        super().__init__(config,rngs)
        self.pure_noise_probability=config.pure_noise_probability

    def compute_loss(self,rng,observation,actions,*,train=False):
        preprocess_key,flow_key=jax.random.split(rng)
        observation=model_module.preprocess_observation(preprocess_key,observation,train=train)
        mixed,target,times=noisy_actions(flow_key,actions,self.pure_noise_probability)
        prefix,prefix_mask,prefix_ar=self.embed_prefix(observation)
        suffix,suffix_mask,suffix_ar,condition=self.embed_suffix(observation,mixed,times)
        mask=jnp.concatenate([prefix_mask,suffix_mask],axis=1)
        attention=pi0.make_attn_mask(mask,jnp.concatenate([prefix_ar,suffix_ar],axis=0))
        positions=jnp.cumsum(mask,axis=1)-1
        (_,suffix_out),_=self.PaliGemma.llm([prefix,suffix],mask=attention,
            positions=positions,adarms_cond=[None,condition])
        velocity=self.action_out_proj(suffix_out[:,-self.action_horizon:])
        # Seven robot action axes; padded model dimensions are not robot labels.
        return jnp.mean(jnp.square(velocity[...,:7]-target[...,:7]),axis=-1)


@dataclasses.dataclass(frozen=True)
class ConditionalPi0Config(pi0_config.Pi0Config):
    pure_noise_probability: float=.5

    def create(self,rng):
        return ConditionalPi0(self,nnx.Rngs(rng))
