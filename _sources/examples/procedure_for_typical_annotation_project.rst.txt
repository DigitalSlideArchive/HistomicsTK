Procedure for managing a typical annotation project
======================================================

Conducting an annotation project requires careful planning and familiarity with
how to manage users, data, and annotations in DSA. Here we describe considerations
in planning for a successful annotation project and provide an overview of steps
to start a project.

While each annotation project is different, the general procedure is quite
similar in various projects. Here we provide a broad overview of a typical
annotation project. This guide is intended for study coordinators and/or
developers who intend to conduct an annotation project with multiple users
for pedagogical or research purposes. Some of these steps can be carried
out completely using HistomicsUI, while others are better handled programmatically
using the API.

**Define the problem**: The structure of an annotation problem is determined by
the end goal. If the goal is to generate data for training of validating machine
learning algorithms then the annotation format will be determined by the
algorithm requirements. If you want to measure inter-rater variability then
you need to plan to collect multiple annotations of the same content. Other
considerations include how to avoid bias in selection of regions for algorithm
validation, and whether and how annotators will receive feedback or make corrections.

**Create user groups**: User groups enable management access permissions for
folders and annotations and should be set up at the beginning of the project
to ensure that permissions can be maintained as people join or leave the project.
When combined with folder permissions, group permissions enable control over who
can create and modify annotations. The `girder permissions model <https://girder.readthedocs.io/en/stable/user-guide.html#permissions>`_ is set up as follows:

- **Folders** inherit permissions from their parent folder by default but their permissions can be altered after creation. If a folder is copied from one parent folder to another, the copy will inherit the permissions of the destination parent folder. If a folder is moved its original permissions will be preserved.
- **Items/slides** do not have their own permissions but inherit the permissions of the folder where they reside.
- **Annotation documents**, by default, inherit permissions from the folder in which their associated item exists. The person who created the annotation document using HistomicsUI or who posted it using the API has owner permissions by default.  These default permissions can be altered after creation. If an item is moved from one folder to another, the annotations will retain their original permissions even while the item permissions will be inherited from the new folder.

In our experience it is helpful to establish three groups to manage permissions for an annotation project: a view group, an edit group, and an owner group. These groups should be created at the beginning of a project before any folders or data are created, ensuring that any folders created, items added, and annotations generated will have these groups available in their permissions structure. For example, if you want to promote a user to owner permissions at the project midpoint, you can simply add them to the owner group and they will have owner access to the entire project. Failure to initially create groups will result in developing code to manually alter the permissions on individual documents via API calls for any changes in permissions.

**Defining groups and styles**: Standardizing group names and their annotation styles are necessary to make review and processing possible. Our `video on using HistomicsUI <https://www.youtube.com/watch?v=HTvLMyKYyGs>`_ shows how to create groups and set their styles, and how to export these to a JSON styles file that can be distributed during a study. Users are asked to import the styles JSON file which is stored locally in their browser cookies. This ensures that all users have consistent group naming and that groups are displayed consistently during review (e.g. tumor nuclei annotations are always shown in red). Below is a sample style JSON:

.. code-block:: javascript

    [
      {
        "lineWidth": 4,
        "lineColor": "rgb(255, 0, 0)",
        "fillColor": "rgba(0, 0, 0, 0)",
        "id": "tumor",
        "label": {
          "value": "tumor"
        },
        "group": "tumor"
      },
      {
        "lineWidth": 4,
        "lineColor": "rgb(0, 0, 255)",
        "fillColor": "rgba(0, 0, 0, 0)",
        "id": "TILs",
        "label": {
          "value": "TILs"
        },
        "group": "TILs"
      },
    ]

**Create an annotation protocol document**: Creating a written protocol instructing users how and what to annotate can improve consistency and reproducibility, and forces participants to consider and discuss project objectives. Here are two sample protocols we have used for our own semantic segmentation and cell detection / classification projects:

- `Semantic segmentation + Nucleus centroids <https://ndownloader.figshare.com/files/22394667>`_: This protocol was developed for an FDA study of tumor-infiltrating lymphocytes in breast cancer.
- `Nucleus detection / classification <https://ndownloader.figshare.com/files/22394670>`_: This protocol describes the process for annotating nuclei in breast cancer.

To provide users with the best experience in your annotation projects we recommend that they:

- Use Google Chrome browser.
- Clear their browser cache if you update the DSA server container.
- Use standardized styles generated by a study coordinator, and re-import these styles if they clear their browser cache.
- Clearly understand the type of annotation elements they should generate (points, rectangles, etc.).

    **Note**: If your protocol specifies polygonal annotations make sure to utilize the polygon tool not the line tool in HistomicsML. Generating mask images from line elements can be difficult due to ambiguities in line element coordinates. When using the polygon tool users should still avoid introducing self-crossings as they trace structures since these introduce issues in annotation processing.

**Create folders, permissions, and assign slides**: Folder permissions allow control over which slides and annotations users can see, and when combined with group permissions can help users collaborate, review, and correct each other if desired. Here are a few project scenarios that require different approaches to folder permissions:

- Each user has their own independent set of slides to annotate from scratch
    Create a folder for each user and copy their slides to this folder. Add each user to their folder with edit access to prevent users from seeing each others’ slides or annotations.
- Multiple users working collaboratively on the same slides
    Create a folder for each group of users who will collaborate. Creating user groups for each of these collaborations may be more sustainable than adding each individual user to the folder permissions.
- A study coordinator will be creating regions of interest (ROI) defining regions where users will annotate
    All project slides are placed in a single folder where the pathologist has edit access to create the ROI annotations. Slides are then moved to the respective user folders. The moved slides will inherit the user folder permissions, while the ROI annotation will retain their original permissions (not editable by the users). Alternatively, you can distribute the slides, draw ROIs within user folders, then programmatically edit the ROI annotation permissions to prevent editing by users.
- Multiple users annotating the same ROI to measure inter-observer variability
    Same procedure as above, except that slides are copied after ROI creation instead of moved, and user folder permissions are set up to blind users from each other.

**Backup data regularly**: The topic `“Local backup and SQL querying of annotation data” <annotation_database_backup_and_sql_parser.ipynb>`_ provides details on how to back up the database. Using the SQLite backup option enables you to conveniently query the database and to monitor annotation progress.

    **Note**: Running out of AssetStore space (disk space) will corrupt your annotation database. If your project involves users uploading whole slide images (WSI), make sure to setup user upload quotas. Keep in mind that a typical WSI is often multiple Gigabytes in size. Backup your annotations often.

**Freeze annotations**: At some point you need to freeze all annotations and create a static version of the database for processing. Freezing annotations can be accomplished by changing group permissions to view.

**Generating galleries for annotation review**: Study coordinators can review and/or correct annotations directly through the HistomicsUI interface. In some studies where annotations are sparsely spread throughout a slide, it may be helpful to create review galleries for quality control and review. More details on this are provided in `“Creating mosaic galleries for rapid annotation review” <creating_gallery_images_review.ipynb>`_.

**Preparing annotations for analysis including machine learning**: HistomicsTK provides tools for converting annotations into mask and label image formats. See `“Converting annotations to semantic segmentation mask images” <annotations_to_semantic_segmentation_masks.ipynb>`_ and `"Converting annotations to object segmentation mask images" <annotations_to_object_segmentation_masks.ipynb>`_ for more details.
