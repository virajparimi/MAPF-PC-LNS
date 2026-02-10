#pragma once
#include <string>

class Instance;
class Solution;

class CBSReportExporter {
  // Writes a report file consumable by MAPF-PC.
  // Typical usage:
  //   CBSReportExporter exporter;
  //   exporter.printStart();
  //   exporter.writeReport(&instance, &solution);
  //   exporter.printSaveStatus();
  //
  // The output file is named:
  //   report_YYYY-MM-DD_HH-MM-SS.txt
  //
  // Example MAPF-PC command:
  //   ./bin/cbs -m sample_input/empty-16-16.map
  //   -a sample_input/report_2023-10-25_22-10-11.txt -s 2 -k 10
 private:
  std::string outputFile_;

 public:
  void printSaveStatus();
  void printStart();
  void writeReport(const Instance* inst, const Solution* sol);
  inline const std::string& getOutputFile() const { return outputFile_; }
};
