

## EVO Tools

### 1 EDB

EDB(*Evo Debug Tool*) is a pure C tool for debugging and do difftest for Software or Hardware:
1. Trace operators' runtime
2. Connect to port and send/receive result tensor_data/model/operator_asm
3. Difftest and cross-valid run result

#### 1.1 Socket

Socket is a way of connecting two nodes on a network with each other:
1. **Server**: One socket listen to the port of an IP.
2. **Client**: The other socket connect to first.

```txt

        Server              Client
    ┌────────────┐      ┌────────────┐
    │  socket()  │      │  socket()  │
    └─────┬──────┘      └─────┬──────┘
    ┌─────┴──────┐      ┌─────┴──────┐
    │   bind()   │      │  connect() │
    └─────┬──────┘      └─────┬──────┘
    ┌─────┴──────┐            │
    │  listen()  │            │
    └─────┬──────┘            │
    ┌─────┴──────┐            │
    │  accept()  │            │
    └─────┬──────┘            │
    ┌─────┴───────────────────┴──────┐
    │   read()    <----    write()   │
    │   write()   ---->    read()    │
    └─────┬───────────────────┬──────┘
    ┌─────┴──────┐      ┌─────┴──────┐
    │   close()  │      │   close()  │
    └────────────┘      └────────────┘

```