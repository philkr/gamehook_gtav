#include <windows.h>
#include <unordered_map>
#include <unordered_set>
#include "log.h"
#include "sdk.h"
#include <iomanip>
#include <chrono>
#include <fstream>
#include <iterator>
#include "scripthook\main.h"
#include "scripthook\natives.h"
#include "util.h"
#include "gtastate.h"
#include "ps_output.h"
#include "vs_static.h"
#include "ps_flow.h"
#include "ps_noflow.h"

uint32_t MAX_UNTRACKED_OBJECT_ID = 1 << 14;

// For debugging, use the current matrices, not the past to estimate the flow
//#define CURRENT_FLOW

double time() {
	auto now = std::chrono::system_clock::now();
	return std::chrono::duration<double>(now.time_since_epoch()).count();
}

struct TrackData: public TrackedFrame::PrivateData {
	struct BoneData {
		float data[255][3][4] = { 0 };
	};
	struct WheelData {
		float4x4 data[2] = { 0 };
	};

	uint32_t id=0, last_frame=0, has_prev_rage=0, has_cur_rage=0;
	float4x4 prev_rage[4] = { 0 }, cur_rage[4] = { 0 };
	std::unordered_map<int, BoneData> prev_bones, cur_bones;
	std::vector<WheelData> prev_wheels, cur_wheels;
	void swap() {
		has_prev_rage = has_cur_rage;
		memcpy(prev_rage, cur_rage, sizeof(prev_rage));
		prev_bones.swap(cur_bones);
		prev_wheels.swap(cur_wheels);
	}
};
struct VehicleTrack {
	float4x4 rage[4];
	float4x4 wheel[16][2];
};
const size_t RAGE_MAT_SIZE = 4 * sizeof(float4x4);
const size_t VEHICLE_SIZE = RAGE_MAT_SIZE;
const size_t WHEEL_SIZE = RAGE_MAT_SIZE + 2 * sizeof(float4x4);
const size_t BONE_MTX_SIZE = 255 * 4 * 3 * sizeof(float);

struct GTA5 : public GameController {
	GTA5() : GameController() {
	}
	virtual bool keyDown(unsigned char key, unsigned char special_status) {
		return false;
	}
	virtual std::vector<ProvidedTarget> providedTargets() const override {
		return { {"albedo"}, {"final"}, { "prev_disp", TargetType::R32_FLOAT, true } };
	}
	virtual std::vector<ProvidedTarget> providedCustomTargets() const {
		// Write the disparity into a custom render target (this name needs to match the injection shader buffer name!)
		return { {"flow_disp", TargetType::R32G32B32A32_FLOAT, true}, { "flow", TargetType::R32G32_FLOAT}, { "disparity", TargetType::R32_FLOAT },{ "occlusion", TargetType::R32_FLOAT }, { "object_id", TargetType::R32_UINT } };
	}
	CBufferVariable rage_matrices = { "rage_matrices", "gWorld", {0}, {4*16*sizeof(float)} }, wheel_matrices = { "matWheelBuffer", "matWheelWorld",{ 0 },{ 32 * sizeof(float) } }, rage_bonemtx = { "rage_bonemtx", "gBoneMtx",{ 0 },{ BONE_MTX_SIZE } };
	enum ObjectType {
		UNKNOWN=0,
		VEHICLE=1,
		WHEEL=2,
		TREE=3,
		PEDESTRIAN=4,
		BONE_MTX=5,
		PLAYER=6,
	};
	std::unordered_map<ShaderHash, ObjectType> object_type;
	std::unordered_set<ShaderHash> final_shader;
	std::shared_ptr<Shader> vs_static_shader = Shader::create(ByteCode(VS_STATIC, VS_STATIC + sizeof(VS_STATIC))),
		ps_output_shader = Shader::create(ByteCode(PS_OUTPUT, PS_OUTPUT + sizeof(PS_OUTPUT)), { { "SV_Target6", "flow_disp" }, { "SV_Target7", "object_id" } }),
		flow_shader = Shader::create(ByteCode(PS_FLOW, PS_FLOW + sizeof(PS_FLOW)), { { "SV_Target0", "flow" },{ "SV_Target1", "disparity" },{ "SV_Target2", "occlusion" } }),
		noflow_shader = Shader::create(ByteCode(PS_NOFLOW, PS_NOFLOW + sizeof(PS_NOFLOW)), { { "SV_Target0", "flow" },{ "SV_Target1", "disparity" },{ "SV_Target2", "occlusion" } });
	std::unordered_set<ShaderHash> int_position = { ShaderHash("d05510b7:0d9c59d0:612cd23a:f75d5ebd"), ShaderHash("c59148c8:e2f1a5ad:649bb9c7:30454a34"), ShaderHash("8a028f64:80694472:4d55d5dd:14c329f2"), ShaderHash("53a6e156:ff48c806:433fc787:c150b034"), ShaderHash("86de7f78:3ecdef51:f1c6f9e3:c5e6338f"), ShaderHash("f37799c8:b304710d:3296b36c:46ea12d4"), ShaderHash("4b031811:b6bf1c7f:ef4cd0c1:56541537") };
	virtual std::shared_ptr<Shader> injectShader(std::shared_ptr<Shader> shader) {
		if (shader->type() == Shader::VERTEX) {
			bool has_rage_matrices = rage_matrices.scan(shader);
			if (has_rage_matrices) {
				bool has_wheel = wheel_matrices.scan(shader);
				bool has_rage_bonemtx = rage_bonemtx.scan(shader);
				ObjectType ot;
				if (has_wheel)
					object_type[shader->hash()] = ot = WHEEL;
				else if (hasCBuffer(shader, "vehicle_globals") && hasCBuffer(shader, "vehicle_damage_locals"))
					object_type[shader->hash()] = ot = VEHICLE;
				else if (hasCBuffer(shader, "trees_common_locals"))
					object_type[shader->hash()] = ot = TREE;
				else if (hasCBuffer(shader, "ped_common_shared_locals"))
					object_type[shader->hash()] = ot = PEDESTRIAN;
				else if (has_rage_bonemtx)
					object_type[shader->hash()] = ot = BONE_MTX;

				bool can_inject = true;
				for (const auto & b : shader->cbuffers())
					if (b.bind_point == 0)
						can_inject = false;
				if (int_position.count(shader->hash()))
					can_inject = false;
				if (can_inject) {
					// Duplicate the shader and copy rage matrices
					auto r = shader->subset({ "SV_Position" });
					r->renameOutput("SV_Position", "PREV_POSITION");
					r->renameCBuffer("rage_matrices", "prev_rage_matrices");
					if (ot == WHEEL)
						r->renameCBuffer("matWheelBuffer", "prev_matWheelBuffer", 5);
					if (ot == PEDESTRIAN || ot == BONE_MTX)
						r->renameCBuffer("rage_bonemtx", "prev_rage_bonemtx", 5);
					// TODO: Handle characters properly
					return r;

					return vs_static_shader;
				}
			}
		}
		if (shader->type() == Shader::PIXEL) {
			// prior to v1.0.1365.1
			if (hasTexture(shader, "BackBufferTexture")) {
				final_shader.insert(shader->hash());
			}
			// v1.0.1365.1 and newer
			if (hasTexture(shader, "SSLRSampler") && hasTexture(shader, "HDRSampler")) {
				// Other candidate textures include "MotionBlurSampler", "BlurSampler", but might depend on graphics settings
				final_shader.insert(shader->hash());
			}
			if (hasCBuffer(shader, "misc_globals")) {
				// Inject the shader output
				return ps_output_shader;
			}
		}
		return nullptr;
	}

	virtual void postProcess(uint32_t frame_id) override {

		if (currentRecordingType() != NONE) {
			// Estimate the projection matrix (or at least a subset of it's values)
			// a 0 0 0
			// 0 b 0 0
			// 0 0 c e
			// 0 0 d 0
#define DOT(a,b,i,j) (a[0][i]*b[0][j] + a[1][i]*b[1][j] + a[2][i]*b[2][j] + a[3][i]*b[3][j])
		// With a bit of math we can see that avg_world_view_proj[:][3] = d * avg_world_view[:][2], below is the least squares estimate for d
			float d = DOT(avg_world_view_proj, avg_world_view, 3, 2) / DOT(avg_world_view, avg_world_view, 2, 2);
			// and avg_world_view_proj[:][2] = c * avg_world_view[:][2] + e * avg_world_view[:][3], below is the least squares estimate for c and e
			float A00 = DOT(avg_world_view, avg_world_view, 2, 2), A10 = DOT(avg_world_view, avg_world_view, 2, 3), A11 = DOT(avg_world_view, avg_world_view, 3, 3);
			float b0 = DOT(avg_world_view, avg_world_view_proj, 2, 2), b1 = DOT(avg_world_view, avg_world_view_proj, 3, 2);

			float det = 1.f / (A00*A11 - A10 * A10);
			float c = det * (b0 * A11 - b1 * A10), e = det * (b1 * A00 - b0 * A10);

			// The camera params either change, or there is a ceratin degree of numerical instability (and fov changes, but setting the camera properly is hard ;( )
			float disp_ab[2] = { 6., 4e-5 }; // { -d / e, -c / d };
			// LOG(INFO) << "disp_ab " << -d / e << " " << -c / d;
			disparity_correction->set((const float*)disp_ab, 2, 0 * sizeof(float));
			bindCBuffer(disparity_correction);

			if (currentRecordingType() == DRAW)
				callPostFx(flow_shader);
			else
				callPostFx(noflow_shader);
		}
	}

	float4x4 avg_world = 0, avg_world_view = 0, avg_world_view_proj = 0, prev_view = 0, prev_view_proj = 0;
	uint8_t main_render_pass = 0;
	uint32_t oid = 1, base_id = 1;
	std::shared_ptr<CBuffer> id_buffer, prev_buffer, prev_wheel_buffer, prev_rage_bonemtx, disparity_correction;
	TrackedFrame * tracker = nullptr;
	std::shared_ptr<TrackData> last_vehicle;
	double start_time;
	uint32_t current_frame_id = 1, wheel_count = 0;

	size_t TS = 0;
	virtual void startFrame(uint32_t frame_id) override {
		start_time = time();

		main_render_pass = 2;
		albedo_output = RenderTargetView();

		if (!id_buffer) id_buffer = createCBuffer("IDBuffer", sizeof(int));
		if (!prev_buffer) prev_buffer = createCBuffer("prev_rage_matrices", 4*sizeof(float4x4));
		if (!prev_wheel_buffer) prev_wheel_buffer = createCBuffer("prev_matWheelBuffer", 4 * sizeof(float4x4));
		if (!prev_rage_bonemtx) prev_rage_bonemtx = createCBuffer("prev_rage_bonemtx", BONE_MTX_SIZE);
		if (!disparity_correction) disparity_correction = createCBuffer("disparity_correction", 2*sizeof(float));
		base_id = oid = 1;
		last_vehicle.reset();
		wheel_count = 0;
		tracker = trackNextFrame();

		avg_world = 0;
		avg_world_view = 0;
		avg_world_view_proj = 0;
		TS = 0;
	}
	virtual void endFrame(uint32_t frame_id) override {
		if (currentRecordingType() != NONE) {
			mul(&prev_view_proj, avg_world.affine_inv(), avg_world_view_proj);
			mul(&prev_view, avg_world.affine_inv(), avg_world_view);
			current_frame_id++;

			// Copy the disparity buffer for occlusion testing
			copyTarget("prev_disp", "disparity");

			LOG(INFO) << "T = " << time() - start_time << "   S = " << TS;
		}
	}
	RenderTargetView albedo_output;
	virtual DrawType startDraw(const DrawInfo & info) override {
		if ((currentRecordingType() != NONE) && info.outputs.size() && info.outputs[0].W == defaultWidth() && info.outputs[0].H == defaultHeight() && info.outputs.size() >= 2 && info.type == DrawInfo::INDEX && info.instances == 0) {
			bool has_rage_matrices = rage_matrices.has(info.vertex_shader);
			ObjectType type = UNKNOWN;
			{
				auto i = object_type.find(info.vertex_shader);
				if (i != object_type.end())
					type = i->second;
			}
			if (has_rage_matrices && main_render_pass > 0) {
				std::shared_ptr<GPUMemory> wp = rage_matrices.fetch(this, info.vertex_shader, info.vs_cbuffers, true);
				if (main_render_pass == 2) {
					// Starting the main render pass
					albedo_output = info.outputs[0];
					main_render_pass = 1;
				}
				if (main_render_pass == 1) {
					uint32_t id = 0;
					if (wp && wp->size() >= 3 * sizeof(float4x4)) {
						// Fetch the rage matrices gWorld, gWorldView, gWorldViewProj
						const float4x4 * rage_mat = (const float4x4 *)wp->data();
						float4x4 prev_rage[4] = { rage_mat[0], rage_mat[1], rage_mat[2], rage_mat[3] };
						mul(&prev_rage[2], rage_mat[0], prev_view_proj);
						mul(&prev_rage[1], rage_mat[0], prev_view);

						// Sum up the world and world_view_proj matrices to later compute the view_proj matrix
						if (type != PEDESTRIAN && type != PLAYER) {
							// There is a 'BUG' (or feature) in GTA V that doesn't draw Franklyn correctly in first person view (rage_mat are wrong)
							add(&avg_world, avg_world, rage_mat[0]);
							add(&avg_world_view, avg_world_view, rage_mat[1]);
							add(&avg_world_view_proj, avg_world_view_proj, rage_mat[2]);
						}

						if (type == WHEEL && last_vehicle) {
							std::shared_ptr<GPUMemory> wm = wheel_matrices.fetch(this, info.vertex_shader, info.vs_cbuffers, true);
							if (wm && wm->size() >= 2 * sizeof(float4x4)) {
								if (last_vehicle->cur_wheels.size() <= wheel_count)
									last_vehicle->cur_wheels.resize(wheel_count + 1);
								memcpy(&last_vehicle->cur_wheels[wheel_count], wm->data(), sizeof(TrackData::WheelData));

								// Set the previous wheel matrix
								if (wheel_count < last_vehicle->prev_wheels.size())
									prev_wheel_buffer->set(last_vehicle->prev_wheels[wheel_count]);
								else
									prev_wheel_buffer->set(last_vehicle->cur_wheels[wheel_count]);
								bindCBuffer(prev_wheel_buffer);
							}
							id = last_vehicle->id;
							wheel_count++;
						} else if (tracker) {
							// Determine the GTA type for search
							TrackedFrame::ObjectType gta_type = TrackedFrame::UNKNOWN;
							if (type == PEDESTRIAN) gta_type = TrackedFrame::PED;
							if (type == VEHICLE) gta_type = TrackedFrame::VEHICLE;

							Vec3f v = { rage_mat[0].d[3][0], rage_mat[0].d[3][1], rage_mat[0].d[3][2] };
							

							TrackedFrame::Object * object;
							// A tighter radius for unknown objects
							if (gta_type == TrackedFrame::UNKNOWN) object = (*tracker)(v, Quaternion::fromMatrix(rage_mat[0]), 0.01f, 0.01f, gta_type);
							else if (gta_type == TrackedFrame::PED) object = (*tracker)(v, Quaternion::fromMatrix(rage_mat[0]), 1.f, 10.f, gta_type);
							else object = (*tracker)(v, Quaternion::fromMatrix(rage_mat[0]), 0.1f, 0.1f, TrackedFrame::UNKNOWN);

							if (object) {
								std::shared_ptr<TrackData> track = std::dynamic_pointer_cast<TrackData>(object->private_data);
								if (!track) // Create a track if the object is new
									object->private_data = track = std::make_shared<TrackData>();
								if (object->type() == TrackedFrame::PLAYER) type = PLAYER;
								// Advance a tracked frame
								if (track->last_frame < current_frame_id) {

									// Advance time
									track->swap();

									// Update the rage matrix
									memcpy(track->cur_rage, rage_mat, sizeof(track->cur_rage));
									track->has_cur_rage = 1;

									if (track->last_frame != current_frame_id - 1) {
										// Avoid objects poping in and out
										track->has_prev_rage = 0;
										track->prev_bones.clear();
										track->prev_wheels.clear();
									}
									track->last_frame = current_frame_id;
								}
								// Update the bone_mtx
								if (type == PEDESTRIAN || type == BONE_MTX) {
									std::shared_ptr<GPUMemory> bm = rage_bonemtx.fetch(this, info.vertex_shader, info.vs_cbuffers, true);
									if (bm) {
										if (!track->cur_bones.count(info.vertex_buffer.id)) {
											memcpy(&track->cur_bones[info.vertex_buffer.id], bm->data(), sizeof(TrackData::BoneData));
											TS += sizeof(TrackData::BoneData);
										} else if (0) {
											if (memcmp(&track->cur_bones[info.vertex_buffer.id], bm->data(), sizeof(TrackData::BoneData))) {
												LOG(WARN) << "Bone matrix changed for object " << info.pixel_shader << " " << info.vertex_shader << " " << info.vertex_buffer.id;
												LOG(INFO) << object->type() << " " << object->id;
												return HIDE;
											}
										}
									}
								}
								// Set the prior projection view
								if (track->has_prev_rage)
									memcpy(prev_rage, &track->prev_rage, sizeof(prev_rage));

								// Set the prior bone mtx
								if (type == PEDESTRIAN || type == BONE_MTX) {
									if (track->prev_bones.count(info.vertex_buffer.id))
										prev_rage_bonemtx->set(track->prev_bones[info.vertex_buffer.id]);
									else if (track->cur_bones.count(info.vertex_buffer.id))
										prev_rage_bonemtx->set(track->cur_bones[info.vertex_buffer.id]);
									bindCBuffer(prev_rage_bonemtx);
								}

								track->id = id = MAX_UNTRACKED_OBJECT_ID + object->id;
								if (type == VEHICLE) {
									last_vehicle = track;
									wheel_count = 0;
								}
							}
							else if (type == PEDESTRIAN) {
								return HIDE;
							}
						}
						prev_buffer->set((float4x4*)prev_rage, 4, 0);
						bindCBuffer(prev_buffer);

						id_buffer->set(id);
						bindCBuffer(id_buffer);
						return RIGID;
					}
				}
			}
		} else if (main_render_pass == 1) {
			// End of the main render pass
			copyTarget("albedo", albedo_output);
			main_render_pass = 0;
		}
		return DEFAULT;
	}
	virtual void endDraw(const DrawInfo & info) override {
		if (final_shader.count(info.pixel_shader)) {
			// Draw the final image (right before the image is distorted)
			copyTarget("final", info.outputs[0]);
		}
	}
	virtual std::string gameState() const override {
		if (tracker)
			return toJSON(tracker->info);
		return "";
	}
	virtual bool stop() { return stopTracker(); }
};
REGISTER_CONTROLLER(GTA5);


BOOL WINAPI DllMain(HINSTANCE hInst, DWORD reason, LPVOID) {
	if (reason == DLL_PROCESS_ATTACH) {
		LOG(INFO) << "GTA5 turned on";
		initGTA5State(hInst);
	}

	if (reason == DLL_PROCESS_DETACH) {
		releaseGTA5State(hInst);
		LOG(INFO) << "GTA5 turned off";
	}
	return TRUE;
}
