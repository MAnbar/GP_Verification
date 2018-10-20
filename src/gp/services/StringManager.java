
package gp.services;


public class StringManager {
    public static void printPadded(String s, int n, boolean endline) {
        if(endline){
            System.out.println(String.format("%1$-" + n + "s", s));  
        }
        else
        {
            System.out.print(String.format("%1$-" + n + "s", s));  
        }
    } 
}
