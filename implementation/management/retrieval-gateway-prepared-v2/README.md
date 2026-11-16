# Fixed-corpus gateway preparation

This isolated candidate connects the native web CLI and authenticated Polar gateway to the existing BrowseComp-Plus corpus service. Production training continues with its frozen source.

## Scope and routing

The driver-owned SessionRegistry metadata selects the retrieval profile. BrowseComp-Plus requires dataset `browsecomp_plus` and profile `browsecomp_plus_fixed`. Model tool arguments cannot override either value. The public-web branch retains its original request contract and 45-second timeout.

The only changed native gateway function is `public_web`. Other native functions and package files match training-v10. The original gateway SHA is `1314e11a86e5ec02eae0b1b3ae757b54e015bcf30d6e5d2056f882c229613f91`. Candidate SHA is `d9142301418091f3690349adc73f3f8688835a8d280400ea975a3dd3406ba194`.

## Preparation and checks

`prepare_gateway.py` clones the pinned training-v10 package into a separate source directory and applies the narrow routing change. It refuses to overwrite an existing candidate. `check_gateway_http.py` uses the actual Polar route and SessionRegistry, three native web CLI subprocesses and the real frozen corpus through temporary loopback services. It requires the existing retrieval runtime and local corpus assets.

Eighteen checks passed. All three CLI exits were integer zero, four corpus request/response receipts matched their saved hashes, the parent process was gone and all three temporary ports were closed. Inference-backend accesses and model/Judge calls were zero. No benchmark question or answer was used. Original R1 failed before tool HTTP because CLI subprocess PYTHONPATH omitted the candidate package. The failed helper and receipt remain. R2 changes the test subprocess environment and passes.

## Remaining work

The evaluation driver still needs to provide the BrowseComp-Plus metadata and use the fixed corpus service. Production gateway activation, an actual model benchmark consumer and full evaluation remain pending. When activating a future gateway, preserve the accepted coordinator queue-timeout repair and accounting/archive behavior. Current C2 source and service remain as originally launched.

The remote SOURCE_PINS manifest freezes the tested candidate package and evidence. This README is a documentation sidecar. ROOT_CPU_HTTP_ACCEPTED_SUMMARY.json records the independent acceptance and its precise CPU transport scope.
