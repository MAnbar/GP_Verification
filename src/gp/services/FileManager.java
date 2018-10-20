package gp.services;

import java.io.File;
import java.util.ArrayList;

public class FileManager {
    
    static public ArrayList<String> checkPath(String pathName){
        File folder = new File(pathName);
        File[] listOfFiles = folder.listFiles();
        ArrayList<String> fileNames=new ArrayList<>();
        
        String fileName;
        for (File file : listOfFiles) {
            if (file.isFile()) {
                fileName=file.getName();
                fileNames.add(fileName);
//                System.out.println("File " + fileName);
            } 
        }
       return fileNames;
    }
}
