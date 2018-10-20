package gp.services;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RunPython {
    public static final String PythonVerificationPath = "C:\\Users\\ElMof\\Documents\\Graduation Project\\JavaPythonTest\\match.py";
    public static final String ImagesPath =  "\"C:/Users/ElMof/Documents/Graduation Project/Images/Input/\"";
    
        public static boolean verifyFace(String imgFilename1, String imgFilename2){
        String[] cmd = new String[3];
                cmd[0] = "python"; // check version of installed python: python -V
        cmd[1] = PythonVerificationPath;
        cmd[2] = ImagesPath+" \""+imgFilename1+"\" \""+imgFilename2+"\"";;
        // create runtime to execute external command
        Runtime rt = Runtime.getRuntime();
        Process pr = null;
        try {
            pr = rt.exec(cmd);
        } catch (IOException ex) {
            Logger.getLogger(RunPython.class.getName()).log(Level.SEVERE, null, ex);
        }

        // retrieve output from python script
        BufferedReader bfr = new BufferedReader(new InputStreamReader(pr.getInputStream()));
        String line = "";
        try {
            line = bfr.readLine();
            if(line.equals("True")){
                return true;
            }
        } catch (IOException ex) {
            Logger.getLogger(RunPython.class.getName()).log(Level.SEVERE, null, ex);
        }
        return false;
    }
}
