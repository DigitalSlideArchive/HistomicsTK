import os

import pooch

registry = {
    #
    # Source: Easy1.png
'Easy1.png': 'sha512:feaf2b24c4ab3123caf1aa35f51acb1d8b83b34941b28d130f878702fd3be4ae9bf46176209f7511d1213da511d414c2b4b7738ad567b089224de9d6c189e664',  # noqa
    #
    # Source: Easy1_cdog_max.npz
'Easy1_cdog_max.npz': 'sha512:59fe9d89509a01f6bd48edb08893c71483874b2c790123d9a80fbf3eb23ce7e04a0dbc188562c974184762164124bc0ce8a45fc6bc3db4a4a8c00e5ea888ab08',  # noqa
    #
    # Source: Easy1_cdog_sigma_max.npz
'Easy1_cdog_sigma_max.npz': 'sha512:749699e66fccad4101d6dc4c28510ff97761167ea1f7762c1ccb4ac7597a7dbc3263728b16e6f3d0c37b5316a1406a6a8e454efb27adafd1a7a3548e37842a7d',  # noqa
    #
    # Source: Easy1_clog_max.npz
'Easy1_clog_max.npz': 'sha512:9c0c5f43192768c792b8d011f0d2e62b32cb2f4b435b4b4e127cdf5a054c37eebd322716dd6e05f2847c3c5e16962d1d0dc6e931a38eaae563ca054831a6f39f',  # noqa
    #
    # Source: Easy1_clog_sigma_max.npz
'Easy1_clog_sigma_max.npz': 'sha512:5ad3e04406133c962025998f4a040bc0d1dc0c7440f5338718e9d7b9408e8690297efad1b398c2ef75d068e92e92b8bf0b59d2214a0301eb29bf97db0824f628',  # noqa
    #
    # Source: Easy1_nuclei_fgnd_mask.npz
'Easy1_nuclei_fgnd_mask.npz': 'sha512:7559357672971ef525857d4f526e26b48af19ae3a63f00cbfdb303be049bd5b2d4d914302b59a0eaadabdee0670381114abed22009f353391a4a2150045776c8',  # noqa
    #
    # Source: Easy1_nuclei_seg_kofahi.npy
'Easy1_nuclei_seg_kofahi.npy': 'sha512:abc098a60e93ed8151d4cfc9991f7ce6ba27d08a22d66e0167bc119ba99addf7a7c4a5f3714bb694b6df53d526d358400632fb7295b4f94ad721da1922ee7794',  # noqa
    #
    # Source: Easy1_nuclei_stain.npz
'Easy1_nuclei_stain.npz': 'sha512:79bb23f71eb4fce6c582e789a4467b8511712c60f44df12e90d9301f9b9fb8163b0926d29ee968c7111f0901bd6e0db97562e53f8f88e2122ba43df561c2d774',  # noqa
    #
    # Source: L1.png
'L1.png': 'sha512:dd7beeac9e02951478213563a74ee9c1bd172137107267c248ec5618f00fd421c9e160ca4565ddfe4589982f35abace0464ab524785f2a22bef991dcb3fab5bf',  # noqa
    #
    # Source: TCGA-06-0129-01Z-00-DX3.bae772ea-dd36-47ec-8185-761989be3cc8.svs
'TCGA-06-0129-01Z-00-DX3.bae772ea-dd36-47ec-8185-761989be3cc8.svs': 'sha512:b7bf2d6a56e90bca599351d93a716a7433fe376b78c8df08fa4b9e3c41b91b879a85dd5597f343bde5a552c0744515685712bf36c5e8b60287b8d991dc304f94',  # noqa
    #
    # Source: TCGA-06-0129-01Z-00-DX3_fgnd_mask_lres.png
'TCGA-06-0129-01Z-00-DX3_fgnd_mask_lres.png': 'sha512:a3e5b63fcdd6d1bd069e508d11eb57acc0b0594aae4be33c2559b4bd7d18e7bee471ae03f65097e404bf71c704691afbb81c4175665403a045b60dbeba431644',  # noqa
    #
    # Source: TCGA-06-0129-01Z-00-DX3_roi_nuclei_bbox.anot
'TCGA-06-0129-01Z-00-DX3_roi_nuclei_bbox.anot': 'sha512:4ff32660a6a6de5877696940b494ba4235a6b232fe4bb457948f40c78db33a3e9df598aa2ea823cbd14656f017ba0866e344bffbc7adeac0af55db30b9d81350',  # noqa
    #
    # Source: TCGA-06-0129-01Z-00-DX3_roi_nuclei_boundary.anot
'TCGA-06-0129-01Z-00-DX3_roi_nuclei_boundary.anot': 'sha512:00de06281b410bb5cf530d7d4fe514f79a1acb475ded333365cb8c42489ceba44d0f7da168a18f7b02c3b6ff40dbde02b5d3c10b9f177afb29a69fd9ec64b49e',  # noqa
    #
    # Source: TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39.svs
'TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39.svs': 'sha512:596872a6180d0562a3db7c1536f159eaa08db36eef544be0c5d578045aaba31998b60445b5b3ac62ce4a02564097b3f587246327f7c97d651a09617596eb0b36',  # noqa
    #
    # Source: TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39.svs_annotations.json
'TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39.svs_annotations.json': 'sha512:19ca12071204d3b8cf9ffeeba6653ef63601d27e42be0693290579f4bf0168f09294ee01580a88f773f2efd576cdadfe91f613cedab4242744c028a38a2c3096',  # noqa
    #
    # Source: TCGA-A2-A0YE-01Z-00-DX1_GET_MergePolygons.svs_annotations.json
'TCGA-A2-A0YE-01Z-00-DX1_GET_MergePolygons.svs_annotations.json': 'sha512:c8d8e714b95a6e40c72d152f2980bc69a40494fd17dfedb2e5a8f2c9afe95a027a6f25c104fddd2fc5cced311f65e55a09498cb6fe1a4154eb34591b2b48a99a',  # noqa
    #
    # Source: sample_svs_image.TCGA-DU-6399-01A-01-TS1.e8eb65de-d63e-42db-af6f-14fefbbdf7bd.svs
'sample_svs_image.TCGA-DU-6399-01A-01-TS1.e8eb65de-d63e-42db-af6f-14fefbbdf7bd.svs': 'sha512:5580c2b5a5360d279d102f1eb5b0e646a4943e362ec1d47f2db01f8e9e52b302e51692171198d0d35c7fa9ec9f5b8e445ef91fa7ea0bdb05ead31ab49e0118f9',  # noqa
}


class DKCPooch(pooch.Pooch):
    def get_url(self, fname):
        self._assert_file_in_registry(fname)
        algo, hashvalue = self.registry[fname].split(':')
        return self.base_url.format(algo=algo, hashvalue=hashvalue)


datastore = DKCPooch(
    path=pooch.utils.cache_location(
        os.path.join(os.environ.get('TOX_WORK_DIR', pooch.utils.os_cache('pooch')), 'externaldata')
    ),
    base_url='https://data.kitware.com/api/v1/file/hashsum/{algo}/{hashvalue}/download',
    registry=registry,
    retry_if_failed=10,
)
