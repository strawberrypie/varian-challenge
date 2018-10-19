# varian-challenge

## api

https://app.swaggerhub.com/apis/b6284/varian/1.0.0

## run server

```
./run_server.py
```

## curl examples

For ping:
```
 curl 127.0.0.1:5000/v1/ping
```

For predict:
```
curl -X POST "127.0.0.1:5000/v1/predict" -H "accept: application/json" -H "Content-Type: application/json" -d "{ \"meta\": { \"name\": \"patient 12345\" }, \"data\": [ \"TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=\" ]}"
```

