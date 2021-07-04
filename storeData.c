/**
 * This is code for an executable that will periodically save usage data for PJM and Comed electrical grids to a csv file.
 * Written by Brandon Pokorny (clickedbigfoot@gmail.com)
 * Compile with the folowing: gcc -lm -Wall storeData.c -o storeData -lcurl
 **/
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <ctype.h>
#include <signal.h>

#define TARGET_URL "https://datasnapshot.pjm.com/content/InstantaneousLoad.aspx"
#define PJM_INDICATOR "<td>PJM RTO Total</td>\r\n\t\t        <td class=\"right\">"
#define PJM_INDICATOR_LEN strlen(PJM_INDICATOR)
#define COMED_INDICATOR "<td>COMED Zone</td>\r\n\t\t        <td class=\"right\">"
#define COMED_INDICATOR_LEN strlen(COMED_INDICATOR)
#define CSV_FILE "usageData.csv"
static int isRunning;

typedef struct WebChunks {
    char *webData;
    size_t size;
} WebChunks;

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
 * Follows prototype function for handling the result of the libcurl operation
 * @param buffer: data that the libcurl returns
 * @param size: size per block of memory in buffer?
 * @param nmemb: number of blocks in the buffer?
 * @param userp: pointer provided by the user. Should be (WebChunks*)
 * @return (size * nmemb)
 **/
size_t write_data(void *buffer, size_t size, size_t nmemb, void *userp) {
    WebChunks *dest = (WebChunks*)userp;
    size_t realSize = size * nmemb;
    dest->webData = realloc(dest->webData, dest->size + realSize);
    if (dest->webData == NULL) {
        perror("realloc failed in function write_data: ");
        exit(1);
    }
    memcpy(dest->webData + dest->size, buffer, realSize);
    dest->size += realSize;
    return realSize;
}

/**
 * Retrieves the webpage for scraping
 * @return pointer to allocated WebChunks struct with the webpage
 **/
WebChunks *getWebPage() {
    //@TODO use libcurl to get webpage
    WebChunks *webpage = (WebChunks*)malloc(sizeof(WebChunks));
    webpage->webData = (char*)malloc(sizeof(char));
    webpage->size = 0;
    CURL *handler = curl_easy_init();
    curl_easy_setopt(handler, CURLOPT_URL, TARGET_URL); //Define behavior of libcurl session
    curl_easy_setopt(handler, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(handler, CURLOPT_WRITEDATA, webpage);
    curl_easy_perform(handler); //Perform libcurl
    curl_easy_cleanup(handler); //Deallocate memory
    return webpage;
}

/**
 * Retrieves the number from the buffer, not including unrelated characters after the number
 * @param scannee: pointer to string from which the numbe ris to be extracted
 * @return the number
 **/
size_t extractNumber(char *scannee) {
    char buffer[256];
    int idx = 0;
    for (char *scanner = scannee; scanner != NULL; scanner++) {
        if (isdigit(*scanner) != 0) {
            buffer[idx] = *scanner;
            idx++;
        }
        else if (*scanner == ',') {
            continue;
        }
        else {
            break;
        }
    }
    buffer[idx] = '\0';
    char *end;
    long num = strtol(buffer, &end, 10);
    if (*end != '\0') {
        perror("Incorrect extraction of number: ");
        exit(1);
    }
    return (int)num;
}

/**
 * Writes to the csv file the intended entry
 * @param f: FILE pointer to csv file
 * @param timeEntry: struct tm* containing the time information for this entry
 * @param pjmUsage: the PJM usage statistic
 * @param comedUsage: the COMED usage statistic
 **/
void writeToCsv(FILE *f, struct tm *timeEntry, int pjmUsage, int comedUsage) {
    char buffer[256];
    buffer[0] = '\0';
    sprintf(buffer, "%i.%i.%i.%i.%i,%i,%i\n", timeEntry->tm_year + 1900, timeEntry->tm_mon + 1, timeEntry->tm_mday, timeEntry->tm_hour, timeEntry->tm_min, pjmUsage, comedUsage);
    fprintf(f, "%s", buffer);
}

/**
 * Gathers the pricing data and stores it in a file
 * @param nextEntryTime: pointer to tm struct determining the intended time for this entry
 **/
void storeData(struct tm *nextEntryTime) {
    WebChunks *webpage = getWebPage();
    int pjmUsage = -1;
    int comedUsage = -1;
    //Extract numbers
    size_t safety = PJM_INDICATOR_LEN > COMED_INDICATOR_LEN ? PJM_INDICATOR_LEN : COMED_INDICATOR_LEN;
    for (int i = 0; i < webpage->size - safety; i++) {
        if (strncmp(webpage->webData + i, PJM_INDICATOR, PJM_INDICATOR_LEN) == 0) {
            pjmUsage = extractNumber(webpage->webData + i + PJM_INDICATOR_LEN);
        }
        if (strncmp(webpage->webData + i, COMED_INDICATOR, COMED_INDICATOR_LEN) == 0) {
            comedUsage = extractNumber(webpage->webData + i + COMED_INDICATOR_LEN);
        }
    }
    if (pjmUsage == -1 || comedUsage == -1) {
        perror("Failed to extract usage statistics: ");
        exit(1);
    }
    //Write them to a csv file
    FILE *f = fopen(CSV_FILE, "a+");
    if (f == NULL) {
        perror("Could not open CSV file: ");
        exit(1);
    }
    writeToCsv(f, nextEntryTime, pjmUsage, comedUsage);
    fclose(f);
    //Deallocate memory
    free(webpage->webData);
    free(webpage);
}

void signalHandler(int sig) {
    isRunning = 0;
}

int main(int argc, char **argv) {
    isRunning = 1;
    //Set up sigaction
    struct sigaction sa;
    sigaction(SIGINT, NULL, &sa); //Load old settings
    sa.sa_handler = signalHandler;
    sigaction(SIGINT, &sa, NULL); //Set new ones
    printf("Note: To shut down program, send SIGINT (Ctrl + c) to the console. Please allow up to five minutes to shut it down due to the app's tendency to sleep between web scrapes.\n");
    //Set up loop
    struct tm nextEntryTimeStack;
	struct tm *nextEntryTime = &nextEntryTimeStack;
    getSecondsLeft(nextEntryTime);
    while (isRunning) {
        sleep(getSecondsLeft(nextEntryTime));
        storeData(nextEntryTime);
    }
}