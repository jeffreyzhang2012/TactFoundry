# Host reception timestamp and sensor coordinate frame.
std_msgs/Header header
string sensor_id
uint32 rows
uint32 columns
string units
# Row-major calibrated taxels; length is rows * columns.
float32[] values
