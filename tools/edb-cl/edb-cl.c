// Client side C program to demonstrate Socket
// programming
#include "linenoise.h"
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define SERVER_IP       "127.0.0.1"
#define SERVER_PORT     0xd1ff

int edb_client_loop() {
    char *line = NULL;
    do {
        int status, valread, client_fd;
        struct sockaddr_in serv_addr;
        char buffer[1024] = {0};
        if ((client_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            printf("Socket creation error \n");
            return -1;
        }
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(SERVER_PORT);
        // Convert IPv4 and IPv6 addresses from text to binary
        // form
        if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
            printf("Invalid address/ Address not supported \n");
            return -1;
        }
        if ((status = connect(client_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr))) < 0) {
            printf("Connection Failed \n");
            return -1;
        }
        // Client Loop
        if(line) {
            send(client_fd, line, strlen(line), 0);
            valread = read(client_fd, buffer, 1024 - 1);  // subtract 1 for the null
            // terminator at the end
            printf("Get: %s\n", buffer);
            if(strcmp(line, "q") == 0) {
                printf("Client quit\n");
                linenoiseFree(line);
                break;
            }
            linenoiseFree(line);
        }
        // closing the connected socket
        close(client_fd);
    } while((line = linenoise("(EDB) ")) != NULL);
    return 0;
}

int main(int argc, char const* argv[]) {
    return edb_client_loop();
}