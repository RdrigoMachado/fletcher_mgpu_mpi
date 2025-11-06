#include "backend.hpp"


void insertSource(const float val, const int iSource, DeviceViewFloat1D dev_pc, DeviceViewFloat1D dev_qc);

void propagate(const int sx, const int sy, const int sz, const int bord,
		       const float dx, const float dy, const float dz, const float dt,
			   DeviceViewFloat1D dev_ch1dxx, DeviceViewFloat1D dev_ch1dyy, DeviceViewFloat1D dev_ch1dzz,
			   DeviceViewFloat1D dev_ch1dxy, DeviceViewFloat1D dev_ch1dyz, DeviceViewFloat1D dev_ch1dxz,
			   DeviceViewFloat1D dev_v2px, DeviceViewFloat1D dev_v2pz, DeviceViewFloat1D dev_v2sz, DeviceViewFloat1D dev_v2pn,
			   DeviceViewFloat1D dev_pp, DeviceViewFloat1D dev_pc,
			   DeviceViewFloat1D dev_qp, DeviceViewFloat1D dev_qc);
