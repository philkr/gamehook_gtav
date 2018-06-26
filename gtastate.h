#pragma once
#include <memory>
#include <unordered_map>
#include "json.h"
#include "util.h"

#ifndef _WINDEF_
struct HINSTANCE__; // Forward or never
typedef HINSTANCE__* HINSTANCE;
typedef HINSTANCE HMODULE;
#endif
void initGTA5State(HMODULE hInstance);
void releaseGTA5State(HMODULE hInstance);


struct GameInfo {
	int time_since_player_hit_vehicle;
	int time_since_player_hit_ped;
	int time_since_player_drove_on_pavement;
	int time_since_player_drove_against_traffic;
	int dead;
	Vec3f position, forward_vector;
	float heading;
	int on_foot, in_vehicle, on_bike, money;
};
TOJSON(GameInfo, time_since_player_hit_vehicle, time_since_player_hit_ped, time_since_player_drove_on_pavement, time_since_player_drove_against_traffic, dead, position, forward_vector, heading, on_foot, in_vehicle, on_bike, money)

// N_OBJECTS Maximum number of frames, needs to be a power of 2
#define N_OBJECTS (1<<13)
struct TrackedFrame {
	enum ObjectType {
		UNKNOWN = 0,
		PED = 1,
		VEHICLE = 2,
		OBJECT = 3,
		PICKUP = 4,
		PLAYER = 5,
	};
	struct PrivateData {
		virtual ~PrivateData();
	};
	struct Object {
		uint32_t id = 0;
		uint32_t age = 0;
		Vec3f p;
		Quaternion q;
		std::shared_ptr<PrivateData> private_data;
		ObjectType type() const;
		uint32_t handle() const;
	};
public:
	friend struct Tracker;
	Object objects[N_OBJECTS];
	NNSearch2D<size_t> object_map;
	void fetch();

public:
	TrackedFrame();
	GameInfo info;
	//Object * operator[](uint32_t id);
	//const Object * operator[](uint32_t id) const;
	Object * operator()(const Vec3f & v, const Quaternion & q);
	Object * operator()(const Vec3f & v, const Quaternion & q, ObjectType t);
	Object * operator()(const Vec3f & v, const Quaternion & q, float D, float QD, ObjectType t);
	const Object * operator()(const Vec3f & v, const Quaternion & q) const;
};

TrackedFrame * trackNextFrame();
bool stopTracker();
