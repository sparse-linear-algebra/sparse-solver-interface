#define SSI_API SSI_EXPORT
#include "sparse_solver_interface.h"

const char* ssi_version(void)
{
  return "0.1.0";
}

ssi_status_t ssi_get_api_v1(ssi_api_v1_t* out_api, size_t out_size)
{
  size_t i;
  unsigned char* bytes;

  if (out_api == 0 || out_size < offsetof(ssi_api_v1_t, reserved)) {
    return SSI_ERROR_INVALID_ARGUMENT;
  }

  bytes = (unsigned char*)out_api;
  for (i = 0; i < out_size; ++i) {
    bytes[i] = 0;
  }

  out_api->size = sizeof(*out_api);
  out_api->abi_version = SSI_ABI_VERSION;
  out_api->capabilities = 0;

  return SSI_SUCCESS;
}
