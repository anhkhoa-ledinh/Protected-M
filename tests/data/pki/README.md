# Test certificates

`a_*` and `b_*` are self-signed pairs (CN localhost, SAN localhost and
127.0.0.1, valid 100 years). They are **test-only and public**: the private
keys are committed on purpose. Never use them on a real server; the server
refuses them unless `--insecure-localhost` is given.

Two pairs exist so a test can ask for "another certificate".
