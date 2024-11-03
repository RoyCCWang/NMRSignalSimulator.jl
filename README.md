# NMRSignalSimulator.jl
Constructs the surrogate and preliminary models for the frequency domain 1D 1H NMR for pulse sequences that are equivalent to the 90 degrees y pulse.

See `/examples/surrogate.jl` for an example.

# Install
From a Julia REPL or script, add the custom registry before adding the package:

```
using Pkg

Pkg.Registry.add(url = "https://github.com/RoyCCWang/RWPublicJuliaRegistry")

Pkt.add("NMRSignalSimulator)
```

# Citation
Our work is undergoing peer review.

# License
This project is licensed under the GPL V3.0 license; see the LICENSE file for details. Individual source files may contain the following tag instead of the full license text:
```
SPDX-License-Identifier: GPL-3.0-only
```

Using SPDX enables machine processing of license information based on the SPDX License Identifiers and makes it easier for developers to see at a glance which license they are dealing with.

