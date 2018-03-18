#ifndef __JOURNAL_H
#define __JOURNAL_H

#include <fstream>
#include <string>


class Journal {
public:
  virtual void WriteHeader() {
  }

  virtual void Write(int update_cycle_num,
		     int particle_id,
		     double px, double py, double theta,
		     double particle_weight,
		     int observation_count,
		     int visible_mark_count) {
  }
};


class JournalFile: public Journal {
public:
  JournalFile(const std::string &file_name): ofs_(file_name, std::ofstream::out) {
  }

  void WriteHeader() {
    ofs_ << "update;pid;x;y;theta;weight;observation_count;visible_mark_count;" << std::endl;
  }

  void Write(int update_cycle_num,
	     int particle_id,
	     double px, double py, double theta,
	     double particle_weight,
	     int observation_count,
	     int visible_mark_count) {
    ofs_ <<
      update_cycle_num << ";" <<
      particle_id << ";" <<
      px << ";" << py << ";" << theta << ";" <<
      particle_weight << ";" <<
      observation_count << ";" <<
      visible_mark_count << ";" << std::endl;
  }

private:
  std::ofstream ofs_;
};


#endif
