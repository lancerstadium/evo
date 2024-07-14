// Client side C program to demonstrate Socket
// programming
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include "linenoise.h"

#ifndef __EDB_CLIENT_H__

#define SERVER_IP       "127.0.0.1"
#define SERVER_PORT     0xd1ff

/**
 * @brief 
 * 
 * @param cmd       Send Command
 * @param buffer    Revice Data
 * @return int 
 * 
 * @note
 * 
 * ```c
 *  char *cmd = NULL;
 *  int ret = 0;
 *  char buffer[1024] = {0};
 *  do {
 *      if(cmd) {
 *          ret = edb_client_send(cmd, buffer);
 *          linenoiseFree(cmd);
 *      }
 *  } while(((cmd = linenoise("(EDB) ")) != NULL) && (ret == 0));
 * ```
 */
static inline int edb_client_send(char *cmd, char *buffer) {
    if(!cmd) return 0;
    int status, valread, client_fd;
    struct sockaddr_in serv_addr;
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
    if(cmd) {
        send(client_fd, cmd, strlen(cmd), 0);
        valread = read(client_fd, buffer, 1024 - 1);  // subtract 1 for the null
        // terminator at the end
        printf("Get: %s\n", buffer);
        if(strcmp(cmd, "q") == 0) {
            printf("Client quit\n");
            return -1;
        }
    }
    // closing the connected socket
    close(client_fd);
    return 0;
}


#endif // __EDB_CLIENT_H__