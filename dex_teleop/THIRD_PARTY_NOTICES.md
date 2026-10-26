# Third-party notices

## MANUS SDK (Python SDK and Integrated SDK)

`dex_teleop.tracking.manus_python_bridge` imports the user-installed official
MANUS Core 3.2 Python SDK (`manus_sdk`, a CFFI binding shipped in the MANUS
SDK package) at runtime, and the optional source-built `native/manus_bridge`
sidecar links against the C++ SDK; both call `CoreSdk_InitializeIntegrated`
or `CoreSdk_InitializeCore`. Neither requires or redistributes the MANUS Core
desktop application. Proprietary MANUS headers, the `manus_sdk` package and
its `libManusSDK` binaries, glove calibration files, and license data are
external inputs: they are not distributed or copied into this repository
(`tests/fake_manus_sdk` is an independent test double that only mirrors the
public function names and enum values). Use of the integration is subject to
the MANUS Software License Agreement supplied with the installed SDK.

Powered by MANUS.

This software contains source code provided by Manus Technology Group B.V.

## VIVE Ultimate Tracker (VIVE Hub for Linux)

`dex_teleop/tracking/vive_ultimate.py` reads tracker poses from the
`tracker_server` daemon of HTC's VIVE Hub for Linux over its Unix domain
socket. The wire format was implemented from HTC's published protocol
description and Python reference client (`tracker_ipc_client_ref.py` in the
`ViveRoboticsSoftware/vive-ultimate-tracker-sdk` repository). No HTC code,
library (`libvut_sdk`, `libultimate-tracker`) or binary is vendored; VIVE Hub
for Linux and the VIVE Ultimate Tracker SDK are proprietary to HTC and
governed by the VIVE Product EULA, and must be installed separately.

## AnyDexRetarget

Parts of `dex_teleop/retargeting`, `dex_teleop/tracking/hts.py`, the hand
configuration YAML files, and the retargeting URDFs are adapted from
AnyDexRetarget.

MIT License

Copyright (c) 2025 Shiquan Qiu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## dex-retargeting

The optional DexPilot backend uses `dex-retargeting==0.4.6` at runtime. The
package is not vendored in this repository.

The MIT License (MIT)

Copyright (c) 2023 Yuzhe Qin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## NVIDIA IsaacTeleop and Isaac Sim references

The packaged Sharpa DexPilot configuration files are adapted from NVIDIA
IsaacTeleop. The tracking-provider boundary was informed by NVIDIA's
`isaacsim.xr.input_devices` extension. Adapted configuration files retain
their upstream copyright and SPDX headers.

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.

SPDX-License-Identifier: Apache-2.0
