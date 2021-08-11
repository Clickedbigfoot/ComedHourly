/**
 * At every five minute mark, this application will scrape the live electricity usage from the PJM and ComEd grids and store
 * that information in a csv file along with the date and time.
 * Then it will determine if a peak will occur within the next hour or not.
 * Created by Brandon Pokorny, clickedbigfoot@gmail.com
 **/

namespace CheckUsage
{
    class Program
    {
        readonly static string DATE_FORMAT = @"yyyy.MM.d.HH.mm";
        readonly static string INSTRUCTIONS = "To shut down program, please press Ctrl + c and wait up to five minutes.";
        readonly static int SECONDS_PER_MINUTE = 60;
        readonly static int MS_PER_SECOND = 1000;
        readonly static string TARGET_URL = @"https://datasnapshot.pjm.com/content/InstantaneousLoad.aspx";
        readonly static string PJM_INDICATOR = "<td>PJM RTO Total</td>\r\n\t\t        <td class=\"right\">";
        readonly static string COMED_INDICATOR = "<td>COMED Zone</td>\r\n\t\t        <td class=\"right\">";
        readonly static string ERROR_PJM_NOT_FOUND = "PJM usage statistic was not found on the website";
        readonly static string ERROR_COMED_NOT_FOUND = "Comed usage statistic was not found on the website";
        readonly static string CSV_FILE = "usageData.csv";
        readonly static string CSV_HEADER = "year.month.day.hour.min,pjmUsage,comedUsage\n";
        readonly static string CONFIG_FILE = "config.txt";
        static bool isRunning;
        static int customInterval;

        /**
         * Determines the milliseconds left until the next entry and also determines what that time will be
         * @param nextEntryTime: DateTime struct reference to set to the exact time under which the next entry will be stored
         * @return the number of milliseconds until the next 5 minute mark on the clock
         **/
        public static int getMillisecondsLeft(ref System.DateTime nextEntry) {
            nextEntry = System.DateTime.Now; //Reset calculation for accuracy
            if (customInterval < 1) {
                //Config file specifies a time interval for each entry
                nextEntry = nextEntry.AddSeconds(customInterval);
                return customInterval * MS_PER_SECOND;
            }
            int secondsLeft = (5 - (nextEntry.Minute % 5)) * SECONDS_PER_MINUTE - nextEntry.Second;
            nextEntry = nextEntry.AddSeconds(-nextEntry.Second);
            nextEntry = nextEntry.AddMinutes(5 - (nextEntry.Minute % 5));
            return (secondsLeft + 3) * MS_PER_SECOND; //Add 3 seconds to ensure that this doesn't calculate 0 seconds the next time it's called
        }

        /**
         * Scrapes the pjm and comed usage statistics from the website
         * @return an array with usages[0] being the pjm statistic and usages[1] being the comed usage statistic
         **/
        public static int[] getStatistics() {
            int[] usages = new int[2];
            System.Net.WebClient wc = new System.Net.WebClient();
            byte[] raw = wc.DownloadData(TARGET_URL);
            string webData = System.Text.Encoding.UTF8.GetString(raw); //Can condense last few lines with wc.DownloadString()
            int idx = webData.IndexOf(PJM_INDICATOR);
            if (idx < 0) {
                throw new System.Exception(ERROR_PJM_NOT_FOUND);
            }
            int length = webData.IndexOf('<', idx + PJM_INDICATOR.Length) - (idx + PJM_INDICATOR.Length);
            string temp = webData.Substring(idx + PJM_INDICATOR.Length, length);
            while (temp.Contains(',')) {
                temp = temp.Replace(",", "");
            }
            usages[0] = System.Int32.Parse(temp);
            idx = webData.IndexOf(COMED_INDICATOR);
            if (idx < 0) {
                throw new System.Exception(ERROR_COMED_NOT_FOUND);
            }
            length = webData.IndexOf('<', idx + COMED_INDICATOR.Length) - (idx + COMED_INDICATOR.Length);
            temp = webData.Substring(idx + COMED_INDICATOR.Length, length);
            while (temp.Contains(',')) {
                temp = temp.Replace(",", "");
            }
            usages[1] = System.Int32.Parse(temp);
            return usages;
        }

        /**
         * Gathers the pricing data and stores it in a file
         * @param nextEntryTime: DateTime struct reference determining the intended time for this entry
         * @param usages: array of the pjm usage statistic and the comed usage statistic, in that order
         * @return the string to store in the csv file. Should be in format DATE,PJM,COMED
         **/
        public static string getEntryText(ref System.DateTime nextEntryTime, int[] usages) {
            return nextEntryTime.ToString(DATE_FORMAT) + "," + usages[0].ToString()+  "," + usages[1].ToString() + "\r\n";
        }

        /**
         * Gathers the pricing data and stores it in a file
         * @param nextEntryTime: DateTime struct reference determining the intended time for this entry
         **/
        public static void storeData(ref System.DateTime nextEntryTime) {
            int[] values = getStatistics();
            string entryText = getEntryText(ref nextEntryTime, values);
            System.IO.File.AppendAllText(CSV_FILE, entryText);
        }

        static void Main(string[] args)
        {
            isRunning = true;
            System.Console.CancelKeyPress += delegate(object sender, System.ConsoleCancelEventArgs e) {
                e.Cancel = true;
                Program.isRunning = false;
            };
            if (System.IO.File.Exists(CONFIG_FILE)) {
                string customIntervalStr = System.IO.File.ReadAllText(CONFIG_FILE).Trim();
                if (customIntervalStr.Length < 1) {
                    customIntervalStr = "-1";
                }
                while (customIntervalStr.Contains(",")) {
                    customIntervalStr = customIntervalStr.Replace(",", "");
                }
                customInterval = (int)(float.Parse(customIntervalStr));
                System.Console.WriteLine("Custom Time Interval: {0} seconds", customInterval);
            }
            else {
                customInterval = -1;
            }
            System.Console.WriteLine(INSTRUCTIONS);
            if (!System.IO.File.Exists(CSV_FILE)) {
                System.IO.File.WriteAllText(CSV_FILE, CSV_HEADER); //Creates file and writes header
            }
            System.DateTime nextEntry = System.DateTime.Now;
            System.Threading.Thread.Sleep(getMillisecondsLeft(ref nextEntry));
            while (isRunning) {
                try {
                    storeData(ref nextEntry);
                }
                catch (System.Exception e) {
                    System.Console.WriteLine("{0} exception caught for entry {1}. Trying again in one minute.", e, nextEntry.ToString(DATE_FORMAT));
                    System.Threading.Thread.Sleep(SECONDS_PER_MINUTE * MS_PER_SECOND);
                    try {
                        storeData(ref nextEntry); //Try 2
                    }
                    catch (System.Exception e2) {
                        System.Console.WriteLine("{0} exception caught for entry {1}. Trying again in one minute.", e2, nextEntry.ToString(DATE_FORMAT));
                        System.Threading.Thread.Sleep(SECONDS_PER_MINUTE * MS_PER_SECOND);
                        try {
                            storeData(ref nextEntry); //Try 3
                        }
                        catch (System.Exception e3) {
                            System.Console.WriteLine("{0} exception caught for entry {1}. Skipping timeslot.", e3, nextEntry.ToString(DATE_FORMAT));
                        }
                    }
                }
                System.Threading.Thread.Sleep(getMillisecondsLeft(ref nextEntry));
            }
            System.Console.WriteLine("Exiting program");
        }
    }
}
