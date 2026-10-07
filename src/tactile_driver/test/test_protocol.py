import pytest
from tactile_driver.protocol import parse_frame


def test_valid_frame():
    assert parse_frame(b'{"values":[1,2.5,-3,0]}\n', 4) == [1.0, 2.5, -3.0, 0.0]


@pytest.mark.parametrize('frame', [
    b'not json', b'[]', b'{}', b'{"values":[1]}',
    b'{"values":[true,2,3,4]}', b'{"values":[NaN,2,3,4]}',
    b'{"values":[1e39,2,3,4]}', b'{"values":["1",2,3,4]}',
])
def test_reject_invalid_frame(frame):
    with pytest.raises(ValueError):
        parse_frame(frame, 4)
