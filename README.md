# FriendlyFit

[![Build Status](https://img.shields.io/travis/guillochon/FriendlyFit.svg)](https://travis-ci.org/guillochon/FriendlyFit)
[![Coverage Status](https://coveralls.io/repos/github/guillochon/FriendlyFit/badge.svg?branch=master)](https://coveralls.io/github/guillochon/FriendlyFit?branch=master)
[![Python Version](https://img.shields.io/badge/python-3.4%2C%203.5-blue.svg)](https://www.python.org)

To run, download an event from the OSC and point FriendlyFit to it in the following way:

```bash
mpirun -np 8 python3.5 -m friendlyfit --event-paths /path/to/file/SN2015bn.json
```

FriendlyFit is currently bare-bones and is not quite functional yet. Stay tuned!
