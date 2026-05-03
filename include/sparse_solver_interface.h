#ifndef SPARSE_SOLVER_INTERFACE_H
#define SPARSE_SOLVER_INTERFACE_H

#ifdef _WIN32
  #ifdef sparse_solver_interface_EXPORTS
    #define ssi_api __declspec(dllexport)
  #else
    #define ssi_api __declspec(dllimport)
  #endif
#else
  #define ssi_api
#endif

#ifdef __cplusplus
extern "C" {
#endif

ssi_api const char *ssi_version(void);

#ifdef __cplusplus
}
#endif

#endif
