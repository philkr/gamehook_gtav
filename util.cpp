#include "util.h"
#include <algorithm>
#include <emmintrin.h>

inline __m128 operator+(const __m128 & a, const __m128 & b) { return _mm_add_ps(a, b); }
inline __m128 operator-(const __m128 & a, const __m128 & b) { return _mm_sub_ps(a, b); }
inline __m128 operator*(const __m128 & a, const __m128 & b) { return _mm_mul_ps(a, b); }
inline __m128 operator/(const __m128 & a, const __m128 & b) { return _mm_div_ps(a, b); }
inline __m128 operator+(const __m128 & a, float b) { return _mm_add_ps(a, _mm_set_ps1(b)); }
inline __m128 operator-(const __m128 & a, float b) { return _mm_sub_ps(a, _mm_set_ps1(b)); }
inline __m128 operator*(const __m128 & a, float b) { return _mm_mul_ps(a, _mm_set_ps1(b)); }
inline __m128 operator/(const __m128 & a, float b) { return _mm_div_ps(a, _mm_set_ps1(b)); }

CBufferVariable::CBufferVariable(const std::string & cbuffer_name, const std::string & variable_name, size_t size) :cbuffer_name(cbuffer_name), variable_name(variable_name) {
	if (size) {
		offset_ = { 0 };
		size_ = { size };
	}
}

CBufferVariable::CBufferVariable(const std::string & cbuffer_name, const std::string & variable_name, const std::vector<size_t> & offset, const std::vector<size_t> & size) :cbuffer_name(cbuffer_name), variable_name(variable_name), offset_(offset), size_(size) {
}

bool CBufferVariable::scan(std::shared_ptr<Shader> s) {
	if (position_hash_.count(s->hash())) return true;
	for (const auto & cb : s->cbuffers())
		if (!cbuffer_name.size() || cb.name == cbuffer_name)
			for (const auto & v : cb.variables)
				if (!variable_name.size() || v.name == variable_name) {
					position_hash_[s->hash()] = { cb.bind_point, v.offset };
					return true;
				}
	return false;
}

bool CBufferVariable::has(const ShaderHash & h) {
	return position_hash_.count(h);
}

std::shared_ptr<GPUMemory> CBufferVariable::fetch(GameController * c, const ShaderHash & h, const std::vector<Buffer> & cbuffers, bool immediate) const {
	auto i = position_hash_.find(h);
	if (size_.size() && i != position_hash_.end() && i->second.bind_point < cbuffers.size()) {
		std::vector<size_t> o = offset_;
		for (auto & j : o) j += i->second.offset;
		return c->readBuffer(cbuffers[i->second.bind_point], o, size_, immediate);
	}
	return std::shared_ptr<GPUMemory>();
}

template<typename T>
bool has(const std::vector<T> & b, const std::string & name) {
	return std::count_if(b.cbegin(), b.cend(), [&name](const T & b) { return b.name == name; });
}

bool hasBuffer(const std::vector<Shader::Buffer> & b, const std::string & name) {
	return has(b, name);
}

bool hasCBuffer(std::shared_ptr<Shader> s, const std::string & name) {
	return has(s->cbuffers(), name);
}

bool hasSBuffer(std::shared_ptr<Shader> s, const std::string & name) {
	return has(s->sbuffers(), name);
}

bool hasTexture(std::shared_ptr<Shader> s, const std::string & name) {
	return has(s->textures(), name);
}

std::ostream & operator<<(std::ostream & s, const float4x4 & f) {
	return s << "[ [" << f.d[0][0] << ", " << f.d[0][1] << ", " << f.d[0][2] << ", " << f.d[0][3] << "], ["
		<< f.d[1][0] << ", " << f.d[1][1] << ", " << f.d[1][2] << ", " << f.d[1][3] << "], ["
		<< f.d[2][0] << ", " << f.d[2][1] << ", " << f.d[2][2] << ", " << f.d[2][3] << "], ["
		<< f.d[3][0] << ", " << f.d[3][1] << ", " << f.d[3][2] << ", " << f.d[3][3] << "] ]";
}

void mul(float4x4 * out, const float4x4 & a, const float4x4 & b) {
	// This can be in place for a but not b!
	for (int i = 0; i < 4; i++) {
		// For once SSE produces easier code
		__m128 r = _mm_set_ps1(0.f);
		for (int j = 0; j < 4; j++)
			r = r + _mm_loadu_ps(b.d[j]) * a.d[i][j];
		_mm_storeu_ps(out->d[i], r);
	}
}

void add(float4x4 * out, const float4x4 & a, const float4x4 & b) {
	for (size_t i = 0; i < 4; i++)
		_mm_storeu_ps(out->d[i], _mm_loadu_ps(a.d[i]) + _mm_loadu_ps(b.d[i]));
}

void div(float4x4 * out, const float4x4 & a, float b) {
	for (size_t i = 0; i < 4; i++)
		_mm_storeu_ps(out->d[i], _mm_loadu_ps(a.d[i]) / b);
}

std::ostream & operator<<(std::ostream & s, const Vec2f & v) {
	return s << "(" << v.x << "," << v.y << ")";
}
std::ostream & operator<<(std::ostream & s, const Vec3f & v) {
	return s << "(" << v.x << "," << v.y << "," << v.z << ")";
}
std::ostream & operator<<(std::ostream & s, const Quaternion & v) {
	return s << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")";
}
std::string toJSON(const Vec2f & v) {
	return "[" + std::to_string(v.x) + "," + std::to_string(v.y) + "]";
}
std::string toJSON(const Vec3f & v) {
	return "[" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + "]";
}
std::string toJSON(const Quaternion & v) {
	return "[" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + "," + std::to_string(v.w) + "]";
}

float4x4::float4x4(float v) {
	for (int k = 0; k < 16; k++)
		((float*)d)[k] = v;
}

float4x4 & float4x4::operator=(float v) {
	for (int k = 0; k < 16; k++)
		((float*)d)[k] = v;
	return *this;
}

float4x4 float4x4::affine_inv() {
	float4x4 r;
	float det = +d[0][0]*(d[1][1]*d[2][2] - d[2][1]*d[1][2])
		       - d[0][1]*(d[1][0]*d[2][2] - d[1][2]*d[2][0])
		       + d[0][2]*(d[1][0]*d[2][1] - d[1][1]*d[2][0]);
	float invdet = 1 / det;
	r.d[0][0] =  (d[1][1]*d[2][2] - d[2][1]*d[1][2])*invdet;
	r.d[0][1] = -(d[0][1]*d[2][2] - d[0][2]*d[2][1])*invdet;
	r.d[0][2] =  (d[0][1]*d[1][2] - d[0][2]*d[1][1])*invdet;
	r.d[1][0] = -(d[1][0]*d[2][2] - d[1][2]*d[2][0])*invdet;
	r.d[1][1] =  (d[0][0]*d[2][2] - d[0][2]*d[2][0])*invdet;
	r.d[1][2] = -(d[0][0]*d[1][2] - d[1][0]*d[0][2])*invdet;
	r.d[2][0] =  (d[1][0]*d[2][1] - d[2][0]*d[1][1])*invdet;
	r.d[2][1] = -(d[0][0]*d[2][1] - d[2][0]*d[0][1])*invdet;
	r.d[2][2] =  (d[0][0]*d[1][1] - d[1][0]*d[0][1])*invdet;
	r.d[3][3] = 1.f / d[3][3];
	r.d[0][3] = -r.d[3][3] * (d[0][3] * r.d[0][0] + d[1][3] * r.d[0][1] + d[2][3] * r.d[0][2]);
	r.d[1][3] = -r.d[3][3] * (d[0][3] * r.d[1][0] + d[1][3] * r.d[1][1] + d[2][3] * r.d[1][2]);
	r.d[2][3] = -r.d[3][3] * (d[0][3] * r.d[2][0] + d[1][3] * r.d[2][1] + d[2][3] * r.d[2][2]);
	r.d[3][0] = -r.d[3][3] * (d[3][0] * r.d[0][0] + d[3][1] * r.d[1][0] + d[3][2] * r.d[2][0]);
	r.d[3][1] = -r.d[3][3] * (d[3][0] * r.d[0][1] + d[3][1] * r.d[1][1] + d[3][2] * r.d[2][1]);
	r.d[3][2] = -r.d[3][3] * (d[3][0] * r.d[0][2] + d[3][1] * r.d[1][2] + d[3][2] * r.d[2][2]);
	return r;
}

float4x4::operator bool() const {
	for (int i = 0; i < 16; i++)
		if (((const float *)d)[i])
			return true;
	return false;
}
