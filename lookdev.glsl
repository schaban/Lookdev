// Proto-material for Houdini
// Author: Sergey Chaban <sergey.chaban@gmail.com>

// Material
uniform sampler2D g_baseMap;
uniform float g_useBaseMap;
uniform vec3 g_baseMapClr;

uniform sampler2D g_specMap;
uniform float g_useSpecMap;
uniform vec3 g_specMapClr;
uniform float g_roughness;
uniform float g_useRoughMap;
uniform sampler2D g_roughMap;
uniform float g_fresnelGain;
uniform float g_fresnelBias;
uniform float g_specFunc;

uniform sampler2D g_normMap;
uniform float g_useNormMap;
uniform float g_normFactor;
uniform float g_flipBitangent;

uniform float g_useOccl;
uniform sampler2D g_occlMap;
uniform vec3 g_occlClr;
uniform float g_occlPwr;

uniform float g_useXlu;
uniform sampler2D g_xluMap;
uniform float g_useXluMap;
uniform vec3 g_xluClr;
uniform float g_xluInt;
uniform float g_xluSpread;

uniform float ogl_alpha_shader;
uniform mat4 glH_InvViewMatrix;

// Lighting
uniform sampler2D g_smpSH;
uniform float g_useSH;
uniform float g_flipDir;
uniform float g_autoDir;
uniform vec3 g_diffClr;
uniform float g_diffInt;
uniform float g_diffS;
uniform float g_useDiffIncd;
uniform vec2 g_diffIncdRange;
uniform float g_diffPwr;
uniform float g_adjDiffTmpr;
uniform float g_diffCoolShift;
uniform float g_diffWarmShift;
uniform vec3 g_reflClr;
uniform float g_reflInt;
uniform float g_reflS;
uniform float g_useReflIncd;
uniform vec2 g_reflIncdRange;
uniform float g_reflPwr;
uniform float g_useSpec;
uniform vec3 g_specClr;
uniform float g_specInt;
uniform float g_specDist;

uniform float g_useHemi;
uniform vec3 g_hemiSky;
uniform vec3 g_hemiGnd;

uniform float g_useEnvMap;
uniform samplerCube g_cubeMap;
uniform float g_useCubeMap;
uniform sampler2D g_panoMap;
uniform float g_flipPanoX;
uniform float g_flipPanoZ;
uniform vec3 g_envMapClr;
uniform float g_envMapInt;
uniform float g_envMapLvl;

// Pattern
uniform float g_useBasePat;
uniform sampler2D g_basePatMap;
uniform vec2 g_basePatOffs;
uniform vec2 g_basePatScl;
uniform vec3 g_basePatClr;
uniform float g_basePatInt;

uniform float g_useSpecPat;
uniform sampler2D g_specPatMap;
uniform vec2 g_specPatOffs;
uniform vec2 g_specPatScl;
uniform vec3 g_specPatClr;
uniform float g_specPatInt;

uniform float g_useNormPat;
uniform sampler2D g_normPatMap;
uniform vec2 g_normPatOffs;
uniform vec2 g_normPatScl;
uniform vec2 g_normPatFactor;

struct MTL_CONTEXT {
	vec3 wpos;
	vec3 wnrm;
	vec3 wtng;
	vec3 wbtn;
	vec3 weye;
	vec3 vdir;
	vec3 rvec;
	vec3 inrm;
	vec4 uv;

	vec4 baseTex;
	vec4 specTex;
	vec3 occl;
};

struct MTL_LIGHT {
	vec3 diff;
	vec3 refl;
	vec3 spec;
	vec3 hdir;
	vec3 dir;
};

struct SH_LIT {
	vec3 diff;
	vec3 refl;
	vec3 dir;
	float specRate;
};


float clrGetLuminance(vec3 clr) {
	vec3 Y709 = vec3(0.212671, 0.71516, 0.072169);
	return dot(clr, Y709);
}

vec3 clrGetTMI(vec3 clr) {
	float r = clr.r;
	float g = clr.g;
	float b = clr.b;
	float t = b - r;
	float m = (r - g*2.0 + b) * 0.5;
	float i = (r + g + b) / 3.0;
	return vec3(t, m, i);
}

vec3 clrSetTMI(vec3 tmi) {
	float t = tmi[0];
	float m = tmi[1];
	float i = tmi[2];
	float r = i - t*0.5 + m/3.0;
	float g = i - m*2.0/3.0;
	float b = i + t*0.5 + m/3.0;
	return vec3(r, g, b);
}


float litCalcSpecRate(vec3 ldir, MTL_CONTEXT ctx) {
	float specRate = 0.0;
	vec3 hvec = normalize(ldir - ctx.vdir);
	vec3 rvec = reflect(ldir, ctx.wnrm);
	float roughMin = 0.000001;
	float rough = max(roughMin, g_roughness);
	if (g_useRoughMap != 0.0) {
		rough *= texture(g_roughMap, ctx.uv.xy).r;
		rough = max(rough, roughMin);
		rough = min(rough, 1.0);
	}
	float rr = rough*rough;
	if (g_specFunc == 0.0) {
		/* Blinn */
		float nh = max(0.0, dot(ctx.wnrm, hvec));
		specRate = pow(nh, max(0.001, 2.0/(rr*rr) - 2.0));
	} else if (g_specFunc == 1.0) {
		/* Phong */
		float vr = max(0.0, dot(ctx.vdir, rvec));
		specRate = pow(vr, max(0.001, 2.0/rr - 2.0)) / 2.0;
	} else if (g_specFunc == 2.0) {
		/* GGX */
		float nl = max(0.0, dot(ctx.wnrm, ldir));
		float nh = max(0.0, dot(ctx.wnrm, hvec));
		float nv = max(0.0, dot(ctx.wnrm, -ctx.vdir));
		float lh = max(0.0, dot(ldir, hvec));
		float vh = max(0.0, dot(-ctx.vdir, hvec));
		float k = rr / 2.0;
		float t1 = nv*(1.0 - k) + k;
		float t2 = nl*(1.0 - k) + k;
		float d = nh*nh*(rr*rr - 1.0) + 1.0;
		d = (rr*rr) / (3.141593 * d*d);
		float vt = (1.0 - vh*vh) / (vh*vh);
		specRate = (nl*d*pow(lh, 5.0)*vt) / (t1*t2);
	}
	return specRate;
}


vec3 shFetch(float ord) {
	float tw = textureSize(g_smpSH, 0).x;
	return textureLod(g_smpSH, vec2((ord + 0.5)/tw, 0.5), 0).rgb;
}

float shGetOrdWgt(float ord, float s, float wscl) {
	return exp((-ord*ord) / (2.0*s)) * wscl;
}

SH_LIT shLit6(MTL_CONTEXT ctx) {
	vec3 wnrm = ctx.wnrm;
	vec3 vdir = ctx.vdir;
	vec3 rvec = ctx.rvec;

	float x = wnrm.x;
	float y = wnrm.y;
	float z = wnrm.z;
	float zz = z*z;

	float rx = rvec.x;
	float ry = rvec.y;
	float rz = rvec.z;
	float rzz = rz*rz;

	float trel = 1.0 - abs(dot(ctx.vdir, ctx.wnrm)); // 0:front .. 1:edge

	float ds = g_diffS;
	if (g_useDiffIncd != 0.0) {
		float dsMin = ds + g_diffIncdRange.x;
		float dsMax = ds + g_diffIncdRange.y;
		ds = mix(dsMin, dsMax, trel);
		ds = max(0.0, ds);
	}

	float rs = g_reflS;
	if (g_useReflIncd != 0.0) {
		float rsMin = rs + g_reflIncdRange.x;
		float rsMax = rs + g_reflIncdRange.y;
		rs = mix(rsMin, rsMax, trel);
		rs = max(0.0, rs);
	}

	float wscl = 3.141592;
	vec2 w1 = exp(vec2(-1.0) / (2.0*vec2(ds, rs))) * wscl;

	vec3 c00 = shFetch(0);
	vec3 d00 = c00 * 0.2820947917738781;
	/* ord=0 -> w = wscl */
	vec3 diff = d00 * wscl;
	vec3 refl = diff;


	vec3 c1_1 = shFetch(1) * -0.48860251190292;
	diff += c1_1 * y * w1.x;
	refl += c1_1 * ry * w1.y;

	vec3 c10 = shFetch(2) * 0.4886025119029199;
	diff += c10 * z * w1.x;
	refl += c10 * rz * w1.y;

	vec3 c11 = shFetch(3) * -0.48860251190292;
	diff += c11 * x * w1.x;
	refl += c11 * rx * w1.y;


	vec4 dw25 = vec4(2, 3, 4, 5);
	dw25 = -dw25*dw25;
	vec4 rw25 = dw25 / vec4(2.0*rs);
	dw25 /= vec4(2.0*ds);
	dw25 = exp(dw25)*wscl;
	rw25 = exp(rw25)*wscl;

	vec2 vx = vec2(x, rx);
	vec2 vy = vec2(y, ry);
	vec2 vz = vec2(z, rz);
	vec2 vzz = vec2(zz, rzz);

	vec2 tmp20 = 0.9461746957575601*vzz + -0.3153915652525201;
	vec3 c20 = shFetch(6);
	diff += c20 * tmp20.x * dw25[0];
	refl += c20 * tmp20.y * rw25[0];

	vec2 tmp30 = vz * (1.865881662950577*vzz + -1.119528997770346);
	vec3 c30 = shFetch(12);
	diff += c30 * tmp30.x * dw25[1];
	refl += c30 * tmp30.y * rw25[1];

	vec2 tmp40 = 1.984313483298443*vz*tmp30 + (-1.006230589874905 * tmp20);
	vec3 c40 = shFetch(20);
	diff += c40 * tmp40.x * dw25[2];
	refl += c40 * tmp40.y * rw25[2];

	vec2 tmp50 = 1.98997487421324*vz*tmp40 + (-1.002853072844814 * tmp30);
	vec3 c50 = shFetch(30);
	diff += c50 * tmp50.x * dw25[3];
	refl += c50 * tmp50.y * rw25[3];

	vec2 tmp2x1 = -1.092548430592079*vz;

	vec3 c2_1 = shFetch(5);
	vec2 tmp2_1 = tmp2x1 * vy;
	diff += c2_1 * tmp2_1.x * dw25[0];
	refl += c2_1 * tmp2_1.y * rw25[0];

	vec3 c21 = shFetch(7);
	vec2 tmp21 = tmp2x1 * vx;
	diff += c21 * tmp21.x * dw25[0];
	refl += c21 * tmp21.y * rw25[0];

	vec2 tmp3x1 = -2.285228997322329*vzz + 0.4570457994644658;

	vec3 c3_1 = shFetch(11);
	vec2 tmp3_1 = tmp3x1 * vy;
	diff += c3_1 * tmp3_1.x * dw25[1];
	refl += c3_1 * tmp3_1.y * rw25[1];

	vec3 c31 = shFetch(13);
	vec2 tmp31 = tmp3x1 * vx;
	diff += c31 * tmp31.x * dw25[1];
	refl += c31 * tmp31.y * rw25[1];

	vec2 tmp4x1 = vz*(-4.683325804901025*vzz + 2.007139630671868);

	vec3 c4_1 = shFetch(19);
	vec2 tmp4_1 = tmp4x1 * vy;
	diff += c4_1 * tmp4_1.x * dw25[2];
	refl += c4_1 * tmp4_1.y * rw25[2];

	vec3 c41 = shFetch(21);
	vec2 tmp41 = tmp4x1 * vx;
	diff += c41 * tmp41.x * dw25[2];
	refl += c41 * tmp41.y * rw25[2];

	vec2 tmp5x1 = 2.03100960115899*vz*-0.48860251190292 + -0.991031208965115*tmp3x1;

	vec3 c5_1 = shFetch(29);
	vec2 tmp5_1 = tmp5x1 * vy;
	diff += c5_1 * tmp5_1.x * dw25[3];
	refl += c5_1 * tmp5_1.y * rw25[3];

	vec3 c51 = shFetch(31);
	vec2 tmp51 = tmp5x1 * vx;
	diff += c51 * tmp51.x * dw25[3];
	refl += c51 * tmp51.y * rw25[3];

	vx = vec2(x*x - y*y, rx*rx - ry*ry);
	vy = vec2(x*y*2.0, rx*ry*2.0);

	vec3 c2_2 = shFetch(4);
	vec2 tmp2_2 = 0.5462742152960395*vy;
	diff += c2_2 * tmp2_2.x * dw25[0];
	refl += c2_2 * tmp2_2.y * rw25[0];

	vec3 c22 = shFetch(8);
	vec2 tmp22 = 0.5462742152960395*vx;
	diff += c22 * tmp22.x * dw25[0];
	refl += c22 * tmp22.y * rw25[0];

	vec2 tmp3x2 = 1.445305721320277*vz;

	vec3 c3_2 = shFetch(10);
	vec2 tmp3_2 = tmp3x2 * vy;
	diff += c3_2 * tmp3_2.x * dw25[1];
	refl += c3_2 * tmp3_2.y * rw25[1];

	vec3 c32 = shFetch(14);
	vec2 tmp32 = tmp3x2 * vx;
	diff += c32 * tmp32.x * dw25[1];
	refl += c32 * tmp32.y * rw25[1];

	vec2 tmp4x2 = 3.31161143515146*vzz + -0.47308734787878;

	vec3 c4_2 = shFetch(18);
	vec2 tmp4_2 = tmp4x2 * vy;
	diff += c4_2 * tmp4_2.x * dw25[2];
	refl += c4_2 * tmp4_2.y * rw25[2];

	vec3 c42 = shFetch(22);
	vec2 tmp42 = tmp4x2 * vx;
	diff += c42 * tmp42.x * dw25[2];
	refl += c42 * tmp42.y * rw25[2];

	vec2 tmp5x2 = vz*(7.190305177459987*vzz + -2.396768392486662);

	vec3 c5_2 = shFetch(28);
	vec2 tmp5_2 = tmp5x2 * vy;
	diff += c5_2 * tmp5_2.x * dw25[3];
	refl += c5_2 * tmp5_2.y * rw25[3];

	vec3 c52 = shFetch(32);
	vec2 tmp52 = tmp5x2 * vx;
	diff += c52 * tmp52.x * dw25[3];
	refl += c52 * tmp52.y * rw25[3];

	vec2 vx0 = vx;
	vec2 vy0 = vy;
	vx = x*vx0 - y*vy0;
	vy = x*vy0 + y*vx0;

	vec2 tmp3x3 = vec2(-0.5900435899266435);

	vec3 c3_3 = shFetch(9);
	vec2 tmp3_3 = tmp3x3 * vy;
	diff += c3_3 * tmp3_3.x * dw25[1];
	refl += c3_3 * tmp3_3.y * rw25[1];

	vec3 c33 = shFetch(15);
	vec2 tmp33 = tmp3x3 * vx;
	diff += c33 * tmp33.x * dw25[1];
	refl += c33 * tmp33.y * rw25[1];

	vec2 tmp4x3 = -1.770130769779931 * vz;

	vec3 c4_3 = shFetch(17);
	vec2 tmp4_3 = tmp4x3 * vy;
	diff += c4_3 * tmp4_3.x * dw25[2];
	refl += c4_3 * tmp4_3.y * rw25[2];

	vec3 c43 = shFetch(23);
	vec2 tmp43 = tmp4x3 * vx;
	diff += c43 * tmp43.x * dw25[2];
	refl += c43 * tmp43.y * rw25[2];

	vec2 tmp5x3 = -4.403144694917254*vzz + 0.4892382994352505;

	vec3 c5_3 = shFetch(27);
	vec2 tmp5_3 = tmp5x3 * vy;
	diff += c5_3 * tmp5_3.x * dw25[3];
	refl += c5_3 * tmp5_3.y * rw25[3];

	vec3 c53 = shFetch(33);
	vec2 tmp53 = tmp5x3 * vx;
	diff += c53 * tmp53.x * dw25[3];
	refl += c53 * tmp53.y * rw25[3];

	vx0 = vx;
	vy0 = vy;
	vx = x*vx0 - y*vy0;
	vy = x*vy0 + y*vx0;

	vec2 tmp4x4 = vec2(0.6258357354491763);

	vec3 c4_4 = shFetch(16);
	vec2 tmp4_4 = tmp4x4 * vy;
	diff += c4_4 * tmp4_4.x * dw25[2];
	refl += c4_4 * tmp4_4.y * rw25[2];

	vec3 c44 = shFetch(24);
	vec2 tmp44 = tmp4x4 * vx;
	diff += c44 * tmp44.x * dw25[2];
	refl += c44 * tmp44.y * rw25[2];

	vec2 tmp5x4 = 2.075662314881041*vz;

	vec3 c5_4 = shFetch(26);
	vec2 tmp5_4 = tmp5x4 * vy;
	diff += c5_4 * tmp5_4.x * dw25[3];
	refl += c5_4 * tmp5_4.y * rw25[3];

	vec3 c54 = shFetch(34);
	vec2 tmp54 = tmp5x4 * vx;
	diff += c54 * tmp54.x * dw25[3];
	refl += c54 * tmp54.y * rw25[3];

	vx0 = vx;
	vy0 = vy;
	vx = x*vx0 - y*vy0;
	vy = x*vy0 + y*vx0;

	vec2 tmp5x5 = vec2(-0.6563820568401703);

	vec3 c5_5 = shFetch(25);
	vec2 tmp5_5 = tmp5x5 * vy;
	diff += c5_5 * tmp5_5.x * dw25[3];
	refl += c5_5 * tmp5_5.y * rw25[3];

	vec3 c55 = shFetch(35);
	vec2 tmp55 = tmp5x5 * vx;
	diff += c55 * tmp55.x * dw25[3];
	refl += c55 * tmp55.y * rw25[3];


	SH_LIT lit;
	lit.diff = max(vec3(0.0), diff);
	lit.refl = max(vec3(0.0), refl);

	if (g_autoDir != 0.0) {
		// z-shifted hack
		vec3 sclY709 = vec3(0.212671, 0.71516, -0.072169);
		lit.dir = normalize(vec3(dot(c11, sclY709), dot(c1_1, sclY709), dot(c10, sclY709)));
	} else {
		lit.dir = normalize(vec3(clrGetLuminance(c11), clrGetLuminance(c1_1), clrGetLuminance(c10)));
	}
	if (g_flipDir != 0.0) lit.dir = -lit.dir;

	vec3 specDir = lit.dir;
	if (g_specDist > 0.0) {
		specDir = normalize(lit.dir*g_specDist - ctx.wpos); // omni
	}

	lit.specRate = litCalcSpecRate(specDir, ctx);

	return lit;
}

vec3 shIrrad(vec3 dir, vec2 adj) {
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;
	float zz = z*z;

	vec3 w = vec3(3.141593, 2.094395, 0.785398);
	w.xy *= adj;

	vec3 c00 = shFetch(0);
	vec3 irrad = c00 * 0.2820947917738781 * w[0];

	vec3 c1_1 = shFetch(1);
	irrad += c1_1 * y*-0.48860251190292 * w[1];

	vec3 c10 = shFetch(2);
	irrad += c10 * z*0.4886025119029199 * w[1];

	vec3 c11 = shFetch(3);
	irrad += c11 * x*-0.48860251190292 * w[1];

	float tx = x*x - y*y;
	float ty = x*y*2.0;

	vec3 c2_2 = shFetch(4);
	irrad += c2_2 * ty*0.5462742152960395 * w[2];

	vec3 c2_1 = shFetch(5);
	irrad += c2_1 * -1.092548430592079*y * w[2];

	vec3 c20 = shFetch(6);
	irrad += c20 * (0.9461746957575601*zz + -0.3153915652525201) * w[2];

	vec3 c21 = shFetch(7);
	irrad += c21 * -1.092548430592079*x * w[2];

	vec3 c22 = shFetch(8);
	irrad += c22 * tx*0.5462742152960395 * w[2];

	return irrad;
}

vec2 shCalcIrradAdj(vec3 wnrm, vec3 vdir, float s) {
	float adjConst = 1.0;
	float adjLin = 1.0;
	if (s > 0.0) {
		float nv = dot(wnrm, -vdir);
		float fs = 1.0 - (1.0 / (2.0 + 0.64*s));
		float gs = 1.0 / (2.2222 + 0.1*s);
		adjConst = fs + 0.5*gs * (1 + 2*acos(nv)/3.141593 - nv);
		adjLin = fs + (fs - 1.0)*(1.0 - nv);
	}
	return vec2(adjConst, adjLin);
}


MTL_CONTEXT mtlInit() {
	MTL_CONTEXT ctx;

	ctx.wpos = fsIn.wpos;
	ctx.wnrm = normalize(fsIn.wnrm);
	ctx.wtng = normalize(fsIn.wtng);
	ctx.wbtn = normalize(cross(ctx.wtng, ctx.wnrm));
	ctx.weye = (glH_InvViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	ctx.vdir = normalize(ctx.wpos - ctx.weye);
	ctx.inrm = ctx.wnrm;

	ctx.uv.xy = fsIn.texcoord0;
	ctx.uv.zw = fsIn.texcoord0;

	ctx.baseTex = vec4(1.0);
	if (g_useBaseMap != 0.0) {
		ctx.baseTex = texture(g_baseMap, ctx.uv.xy);
	}
	if (g_useBasePat != 0.0) {
		ctx.baseTex *= texture(g_basePatMap, (ctx.uv.xy + g_basePatOffs) * g_basePatScl) * vec4(g_basePatClr * g_basePatInt, 1.0);
	}
	ctx.baseTex.rgb *= g_baseMapClr;

	ctx.specTex = vec4(1.0);
	if (g_useSpecMap != 0.0) {
		ctx.specTex = texture(g_specMap, ctx.uv.xy);
	}
	if (g_useSpecPat != 0.0) {
		ctx.specTex *= texture(g_specPatMap, (ctx.uv.xy + g_specPatOffs) * g_specPatScl) * vec4(g_specPatClr * g_specPatInt, 1.0);
	}
	ctx.specTex.rgb *= g_specMapClr;

	if (g_useNormMap != 0.0) {
		vec4 normTex = texture(g_normMap, ctx.uv.xy);
		vec2 nxy = (normTex.xy + vec2(-0.5)) * 2.0;
		nxy *= g_normFactor;
		if (g_useNormPat != 0.0) {
			vec4 npat = texture(g_normPatMap, (ctx.uv.xy + g_normPatOffs) * g_normPatScl);
			npat.xy = (npat.xy + vec2(-0.5)) * 2.0;
			npat.xy *= g_normPatFactor;
			nxy += npat.xy;
		}
		vec2 sqxy = nxy * nxy;
		float nz = sqrt(max(0.0, 1.0 - sqxy.x - sqxy.y));
		float nflip = g_flipBitangent != 0.0 ? -1.0 : 1.0;
		vec3 nnrm = normalize(nxy.x*ctx.wtng + nxy.y*ctx.wbtn*nflip + nz*ctx.wnrm);
		ctx.wnrm = nnrm;
	}

	ctx.rvec = reflect(ctx.vdir, ctx.wnrm);

	ctx.occl = vec3(1.0);
	if (g_useOccl != 0.0) {
		vec4 occlTex = texture(g_baseMap, ctx.uv.xy);
		ctx.occl = occlTex.rgb;
		ctx.occl *= g_occlClr;
		ctx.occl = pow(ctx.occl, vec3(g_occlPwr));
	}

	return ctx;
}

float mtlFresnel(MTL_CONTEXT ctx) {
	float tcos = dot(ctx.vdir, ctx.wnrm);
	float frpwr = 5.0;
	float fr = clamp(pow(1.0 - tcos, frpwr)*g_fresnelGain + g_fresnelBias, 0.0, 1.0);
	return 1.0 - fr;
}

vec4 mtlPanorama(vec3 dir, float lvl) {
	if (g_flipPanoX != 0.0) dir.x = -dir.x;
	if (g_flipPanoZ != 0.0) dir.z = -dir.z;
	vec2 uv = vec2(0.0);
	float lxz = sqrt(dir.x*dir.x + dir.z*dir.z);
	if (lxz > 1.0e-5) uv.x = -dir.x / lxz;
	uv.y = -dir.y;
	uv = clamp(uv, -1.0, 1.0);
	uv = acos(uv) / 3.141592653;
	uv.x *= 0.5;
	if (dir.z >= 0.0) uv.x = 1.0 - uv.x;
	return textureLod(g_panoMap, uv, lvl);
}

vec3 mtlAdjColorTemperature(MTL_CONTEXT ctx, MTL_LIGHT lit, vec3 clr, float coolShift, float warmShift) {
	vec3 tmi = clrGetTMI(clr);
	float t = tmi[0];
	float d = dot(lit.dir, ctx.wnrm);
	if (d < 0.0) {
		t += mix(0.0, coolShift, -d);
	} else {
		t -= mix(0.0, warmShift, d);
	}
	tmi[0] = t;
	vec3 c = clrSetTMI(tmi);
	return c;
}

MTL_LIGHT mtlCalcLight(MTL_CONTEXT ctx) {
	MTL_LIGHT lit;
	lit.diff = vec3(0.0);
	lit.refl = vec3(0.0);
	lit.spec = vec3(0.0);
	lit.hdir = -ctx.vdir;
	lit.dir = lit.hdir;

	float fresnel = mtlFresnel(ctx);

	if (g_useSH != 0.0) {
		SH_LIT shl = shLit6(ctx);
		lit.diff += pow(shl.diff * g_diffInt, vec3(g_diffPwr)) * g_diffClr;
		lit.refl += pow(shl.refl * g_reflInt * fresnel, vec3(g_reflPwr)) * g_reflClr;
		if (g_useSpec != 0.0) {
			lit.spec += shl.specRate * g_specClr*g_specInt * fresnel;
		}

		if (g_useXlu != 0.0) {
			vec2 adj = shCalcIrradAdj(ctx.wnrm, ctx.vdir, g_xluSpread);
			vec3 xlu = shIrrad(-ctx.wnrm, adj);
			xlu *= g_xluClr * g_xluInt;
			if (g_useXluMap != 0.0) {
				xlu *= texture(g_xluMap, ctx.uv.xy).rgb;
			}
			lit.diff += xlu;
		}

		lit.dir = shl.dir;
	}

	if (g_useHemi != 0.0) {
		float hemiRate = (ctx.wnrm.y + 1.0) * 0.5;
		lit.diff += mix(g_hemiGnd, g_hemiSky, hemiRate);
	}

	if (g_adjDiffTmpr != 0.0) {
		lit.diff = mtlAdjColorTemperature(ctx, lit, lit.diff, g_diffCoolShift, g_diffWarmShift);
	}

	if (g_useEnvMap != 0.0) {
		vec4 reflTex = vec4(0.0);
		if (g_useCubeMap != 0.0) {
			reflTex = textureLod(g_cubeMap, ctx.rvec, g_envMapLvl);
		} else {
			reflTex = mtlPanorama(ctx.rvec, g_envMapLvl);
		}
		lit.refl += reflTex.rgb * g_envMapClr * g_envMapInt * fresnel;
	}

	return lit;
}

vec4 mtlEval(MTL_CONTEXT ctx) {
	MTL_LIGHT lit = mtlCalcLight(ctx);
	vec3 diff = lit.diff * ctx.baseTex.rgb * ctx.occl;
	vec3 spec = (lit.spec + lit.refl) * ctx.specTex.rgb * ctx.occl;
	vec4 clr;
	clr.rgb = diff + spec;
	clr.a = ctx.baseTex.a;
	return clr;
}
