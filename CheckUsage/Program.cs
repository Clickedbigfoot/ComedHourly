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
        static bool isRunning;

        /**
         * Determines the milliseconds left until the next entry and also determines what that time will be
         * @param nextEntryTime: DateTime struct reference to set to the exact time under which the next entry will be stored
         * @return the number of milliseconds until the next 5 minute mark on the clock
         **/
        public static int getMillisecondsLeft(ref System.DateTime nextEntry) {
            nextEntry = System.DateTime.Now; //Reset calculation for accuracy
            int secondsLeft = (5 - (nextEntry.Minute % 5)) * SECONDS_PER_MINUTE - nextEntry.Second;
            nextEntry = nextEntry.AddSeconds(-nextEntry.Second);
            nextEntry = nextEntry.AddMinutes(5 - (nextEntry.Minute % 5));
            return secondsLeft * MS_PER_SECOND;
        }

        static void Main(string[] args)
        {
            isRunning = true;
            System.Console.CancelKeyPress += delegate(object sender, System.ConsoleCancelEventArgs e) {
                e.Cancel = true;
                Program.isRunning = false;
            };
            System.Console.WriteLine(INSTRUCTIONS);
            System.DateTime nextEntry = System.DateTime.Now;
            System.Threading.Thread.Sleep(getMillisecondsLeft(ref nextEntry));
            while (isRunning) {
                System.Console.WriteLine("Local date and time is {0}", nextEntry.ToString(DATE_FORMAT));
                System.Threading.Thread.Sleep(getMillisecondsLeft(ref nextEntry));
            }
            System.Console.WriteLine("Exiting program");
        }
    }
}
