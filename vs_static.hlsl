cbuffer prev_rage_matrices: register(b0) {
	row_major float4x4 prev_worldViewProj : packoffset(c0.x);
};

void main(in float3 pos: POSITION, out float4 prev_pos : PREV_POSITION) {
	prev_pos = mul(float4(pos, 1), prev_worldViewProj);
}
