GFPGAN by TencentARC, official pretrained external baseline.
Source commit: 7552a7791caad982045a7bbe5634bbf1cd5c8679
https://github.com/TencentARC/GFPGAN

Local changes: removed BasicSR registry decorators and replaced only the weight initializer dependency. All model tensors are overwritten by strict-loading the official checkpoint. Full upstream license is included.
