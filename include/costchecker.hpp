#pragma once
#include <string>

class Instance;
class Solution;

class SaveToTxt {
  /*
    Please add  - 

    1) #include "cbs_pc_cost_checker.h" to lns.cpp @lns.cpp::5

    2) to lns.cpp::281
        SavetoTxt sv;
        sv.printStart();
        sv.run_data(&instance_, &solution_);
        sv.filesave();
    
    Description - 
        Simple function to save the instance data such as agent task assignment, start locations,
        task locations and global precedence constraints in a txt format for ingestion to 
        MAPF-PC code. 

        The text file is saved with a timestamp as a unique identifier. 

        Please copy the text file to the appropriate location in the MAPF-PC codebase.

        The corresponding MAPF-PC command (new report names avoid ':' so escaping isn't needed):
        ./bin/cbs -m sample_input/empty-16-16.map -a sample_input/report_2023-10-25_22-10-11.txt -s 2 -k 10
 */
 public:
  std::string outputFile;
  void fileSave();
  void printStart();
  void runData(const Instance* inst, const Solution* sol);
};
