/**
 * This is code for an executable that will periodically save 
 **/
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

// "https://datasnapshot.pjm.com/content/InstantaneousLoad.aspx"
static char *HOSTNAME = "datasnapshot.pjm.com";
//static char *PAGE = "content/InstantaneousLoad.aspx";
//static char *HOSTNAME = "google.com";
static char *GET_REQUEST = "GET /content/InstantaneousLoad.aspx HTTP/1.1\r\nHost: datasnapshot.pjm.com\r\nUser-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0\r\nAccept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8\r\nAccept-Language: en-US,en;q=0.5\r\nAccept-Encoding: gzip, deflate, br\r\nConnection: keep-alive\r\nUpgrade-Insecure-Requests: 1\r\n\r\n";//"GET /content/InstantaneousLoad.aspx HTTP/1.1\r\nHost: datasnapshot.pjm.com\r\nUser-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0\r\nAccept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8\r\nAccept-Language: en-US,en;q=0.5\r\nAccept-Encoding: gzip, deflate, br\r\n\r\n";

/**
 * Determines the seconds left until the next entry and also determines what that time will be
 * @param nextEntryTime: tm struct pointer to set to the exact time under which the next entry will be stored
 * @return the seconds that must be slept until the next entry
 **/
int getSecondsLeft(struct tm *nextEntryTime) {
    time_t rawtime;
    struct tm *timeInfo;
    time(&rawtime);
    timeInfo = localtime(&rawtime);
    int secondsTarget = (5 -(timeInfo->tm_min % 5)) * 60;
    *nextEntryTime = *timeInfo;
    nextEntryTime->tm_sec = 0;
    nextEntryTime->tm_min += 5 - timeInfo->tm_min % 5;
    mktime(nextEntryTime);
    return secondsTarget - timeInfo->tm_sec;
}

/**
 * Testing function that simply prints the current time
 **/
void printTime() {
    time_t rawtime;
    struct tm *timeInfo;
    time(&rawtime);
    timeInfo = localtime(&rawtime);
    printf("Time: %s", asctime(timeInfo));
}

/**
 * Creates a newly allocated string serving as the GET request
 **/

/**
 * Retrieves the PMD usage statistics website
 * @return the PMD usage statistics website as a newly allocated string
 **/
char *getWebPage() {
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET; //AF_INET for IPv4 addresses
    hints.ai_socktype = SOCK_STREAM; //TCP
    int s = getaddrinfo(HOSTNAME, "80", &hints, &result);
    if (s != 0) {
        //Issue with getaddrinfo
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
        exit(1);
    }
    int sock_fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (sock_fd == -1) {
        //Error in making socket
        perror("socket failed");
        exit(1);
    }
    int connect_result = connect(sock_fd, result->ai_addr, result->ai_addrlen);
    if (connect_result == -1) {
        //Issue with connecting
        perror("connect failed");
        exit(1);
    }
    //#define MSG "GET / HTTP/1.0\r\nHOST: google.com \r\n\r\n" //This sequence of four terminating bytes mean the request is finished
    write(sock_fd, GET_REQUEST, strlen(GET_REQUEST)); //TODO wrap this to handle partial writes and EINTR, etc.
    char buffer[1024];
    ssize_t bytesRead = read(sock_fd, buffer, sizeof(buffer));
    while (bytesRead > 0) {
        printf("\nIteration\n\n");
        write(1, buffer, bytesRead);
        printf("Bytes read: %zd\n", bytesRead);
        bytesRead = read(sock_fd, buffer, sizeof(buffer));
    }
    return (char*)malloc(1024);
}

/**
 * Gathers the pricing data and stores it in a file
 * @param nextEntryTime: pointer to tm struct determining the intended time for this entry
 **/
void storeData(struct tm *nextEntryTime) {
    printf("Entry: %s", asctime(nextEntryTime));
    char *webPage = getWebPage();
    //@TODO extrac the PMD and COMED usages
    //@TODO write them to a csv file
    //Deallocate memory
    free(webPage);
}

int main(int argc, char **argv) {
    struct tm nextEntryTimeStack;
	struct tm *nextEntryTime = &nextEntryTimeStack;
    getSecondsLeft(nextEntryTime); storeData(nextEntryTime); exit(0); //Line only for testing purposes
    while (1) {
        sleep(getSecondsLeft(nextEntryTime));
        printTime();
        storeData(nextEntryTime);
    }
}