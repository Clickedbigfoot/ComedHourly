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
        readonly static string DATE_FORMAT = @"yyyy.MMM.d.HH.mm.ss";
        readonly static int SECONDS_PER_MINUTE = 60;

        /**
         * Determines the seconds left until the next entry and also determines what that time will be
         * @param nextEntryTime: DateTime struct reference to set to the exact time under which the next entry will be stored
         * @return the number of seconds until the next 5 minute mark on the clock
         **/
        public static int getSecondsLeft(ref System.DateTime nextEntry) {
            int secondsLeft = (5 - (nextEntry.Minute % 5)) * SECONDS_PER_MINUTE - nextEntry.Second;
            nextEntry = nextEntry.AddSeconds(secondsLeft);
            return secondsLeft;
        }

        static void Main(string[] args)
        {
            System.DateTime nextEntry = System.DateTime.Now;
            System.Console.WriteLine("Local date and time is {0}", nextEntry.ToString(DATE_FORMAT));
            System.Console.WriteLine("Seconds left: {0}", getSecondsLeft(ref nextEntry).ToString());
            System.Console.WriteLine("Local date and time is {0}", nextEntry.ToString(DATE_FORMAT));
        }
    }
}
