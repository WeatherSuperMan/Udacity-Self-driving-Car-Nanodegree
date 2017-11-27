#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
	auto found_null = s.find("null");
	auto b1 = s.find_first_of("[");
	auto b2 = s.find_last_of("]");
	if (found_null != std::string::npos) {
		return "";
	}
	else if (b1 != std::string::npos && b2 != std::string::npos) {
		return s.substr(b1, b2 - b1 + 1);
	}
	return "";
}

int main()
{
	uWS::Hub h;

	PID pid;
	// Initialize the pid variable.
	// TODO: Tweak initial Kp, Ki, Kd values.

	//* p ------> Proportional Hyperparameter
	//* d------> derivative Hyperparameter
	//* i------> integral Hyperparameter-------> need to be very small to start with
	
	
	//pid.Init(0.5, 0.0, 0); 
	// Try 1: start with a simple P controller. 
	//Finished the Track,but only just though, as there were a lot of oscillations!

	
	//pid.Init(0.5, 0.0, 1.0);
	// Try 2: add. some D parameter.
	//Finished the track safely, there are now much less oscillation


	//pid.Init(0.5, 0.0, 10.0);
	// Try 3:  add even more D parameter.
	// The car drove aroudn the track rather smoothly, however, some noticeable spikes in
	// the values of turning angles are observed.

	//pid.Init(0.5, 0.0, 5.0);
	// Try 4: Turn down the D parameter a bit.
	// no noticeable improvement from the try 3


	//pid.Init(1.0, 0.0, 5.0);
	// Try 5: increase the P parameter to 1
	// no noticeable improvement from try 3 or 4

	//pid.Init(0.1, 0.0, 5.0);
	// Try 6: decerase the P parameter
	// significant improvement! The best result ever so far, oscillation has pretty much vanished!


	//pid.Init(0.0, 0.0, 5.0);
	// Try 7: I'm being stupid here, setting p parameter to 0 resulted in the veering off the track straightaway.

	//pid.Init(0.1, 0.01, 5.0);
	// Try 8: Car was off the track again due to high I parameter

	//pid.Init(0.1, -0.0001, 5.0);
	// Try 9: Car off the track


	pid.Init(0.1, 0.0, 5.0);
	// My final try 

	


	h.onMessage([&pid](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
		// "42" at the start of the message means there's a websocket message event.
		// The 4 signifies a websocket message
		// The 2 signifies a websocket event
		if (length && length > 2 && data[0] == '4' && data[1] == '2')
		{
			auto s = hasData(std::string(data));
			if (s != "") {
				auto j = json::parse(s);
				std::string event = j[0].get<std::string>();
				if (event == "telemetry") {
					// j[1] is the data JSON object
					double cte = std::stod(j[1]["cte"].get<std::string>());
					double speed = std::stod(j[1]["speed"].get<std::string>());
					double angle = std::stod(j[1]["steering_angle"].get<std::string>());
					double steer_value;
					/*
					* TODO: Calcuate steering value here, remember the steering value is
					* [-1, 1].
					* NOTE: Feel free to play around with the throttle and speed. Maybe use
					* another PID controller to control the speed!
					*/
				
					pid.UpdateError(cte);
					steer_value = pid.TotalError();
					std::cout << "Steer: " << steer_value << std::endl;
					if (steer_value > 1) {
						steer_value = 1;
					}
					else if (steer_value < -1) {
						steer_value = -1;
					}

					// DEBUG
					std::cout << "CTE: " << cte << " Steering Value: " << steer_value << std::endl;

					json msgJson;

					msgJson["steering_angle"] = steer_value;


					// speed and throttle control

					if (speed <= 15) {
						msgJson["throttle"] = 1.5;

					}
					else if (speed > 15 && speed < 30) {
						msgJson["throttle"] = 0.2;
					}
					else if (speed >= 30) {
						msgJson["throttle"] = 0.0;
					}
					else if (fabs(steer_value) >= 0.5) {
						msgJson["throttle"] = -0.8;
					}

					

					auto msg = "42[\"steer\"," + msgJson.dump() + "]";
					std::cout << msg << std::endl;
					ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
				}
			}
			else {
				// Manual driving
				std::string msg = "42[\"manual\",{}]";
				ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
			}
		}
	});

	// We don't need this since we're not using HTTP but if it's removed the program
	// doesn't compile :-(
	h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
		const std::string s = "<h1>Hello world!</h1>";
		if (req.getUrl().valueLength == 1)
		{
			res->end(s.data(), s.length());
		}
		else
		{
			// i guess this should be done more gracefully?
			res->end(nullptr, 0);
		}
	});

	h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
		std::cout << "Connected!!!" << std::endl;
	});

	h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
		ws.close();
		std::cout << "Disconnected" << std::endl;
	});

	int port = 4567;
	if (h.listen(port))
	{
		std::cout << "Listening to port " << port << std::endl;
	}
	else
	{
		std::cerr << "Failed to listen to port" << std::endl;
		return -1;
	}
	h.run();
}