#!/bin/bash
openssl req -x509 -newkey rsa:4096 -nodes \
    -out cert.pem -keyout key.pem \
    -days 365 -subj "/CN=jenna-ai.local"
    
mkdir -p deployment/security/certs
mv cert.pem key.pem deployment/security/certs/