# Stairs depth diagnostic (depth12)

This is a bounded real MuJoCo v3 probe. It uses no scripted motion or force. The geometry is explicit macro demonstration geometry and is not experimentally supported. It reuses the 65-degree tilt case and changes stair depth to 0.12 m; environment friction is 0.8 and coil self-friction is 0.05; support thresholds are unchanged.

- status: `completed`; wall time: `491.906 s`; warning/reset/non-finite: `False`
- segments: `96`; dt: `5e-05`; duration: `2.4`; sampled frames: `121`
- raw movement classification: `flip`; max penetration: `0.0005473750576079836 m`
- model version: `helical-box-cable-v3`; initial contacts: `0`; initial geometry: `lower material endpoint is positioned near the top tread and upper material endpoint crosses the top tread edge in +x`
- requested/observed world angular velocity: `[0.0, 0.0, 0.0]` / `[0.0, 0.0, 0.0]`; error norm: `0.0`
- candidate flip count: `3` ['last', 'first', 'last']
- confirmed flip count: `3` ['last', 'first', 'last']
- verified steps: `[1, 2, 3]`; first-touch order: `[(1, 'last'), (2, 'first'), (3, 'middle'), (4, 'middle')]`
- stair_0 sampled contacts: `29` frames; first sample `0.02000000000000006`; max sampled normal force `0.48344221164122625` N
- stair_0 samples are reported for initial-top-tread geometry only and are not used as gait dwell or verified-step evidence; gait starts at stair_step >= 1.
- energy at final frame: kinetic `0.007662589603205901`, gravitational `0.05672150756065069`, elastic estimate `0.004491955241154892`, contact work `-0.04797911679177388`

| step | first touch | first sustained | endpoint support | ambiguous | verified |
|---:|---|---|---|---:|---:|
| 1 | last @ 0.3708499999999673 | last @ 0.6493499999999367 | last | False | True |
| 2 | first @ 0.8596999999999135 | first @ 1.0566000000000175 | first | False | True |
| 3 | middle @ 1.6324500000012327 | last @ 1.9886000000019843 | last | False | True |
| 4 | middle @ 2.3576500000011746 | None @ None | None | False | False |

`summary.json` and `support-diagnostics.json` preserve the substep support transitions and intervals. `frames.json` preserves sampled `contact_details` with contact force, geom IDs, stair index, material index, and tread/other surface classification.
