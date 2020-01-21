Annotation mongo database backup and sqlite parser
====================================================

**Overview:**

This utility function allows backup of a single folder (and its subfolders, recursively) from a girder database locally by recursively pulling folder and item (slide) information +/- the annotations. The information is optionally saved as JSON files and/or entries in an SQLite database. Additionally, the annotations may be parsed into tabular format for easy use and querying and saved as csv files and/or entries in the same SQLite database. The advantage of saving things into SQLite format is that they can be easily queried, and this provides a simple tool for managing projects by periodically using queries to monitor progress, for example. 

The SQLite database can easily be viewed using, for example, an offline sqlite viewer like: https://sqlitebrowser.org/dl/ or even an online sqlite viewer like https://sqliteonline.com/ .

The resultant database has four tables (or two if the user decided not to parse things into annotations):

* **folders**: all girder folders contained within the folder that the user wants to backup. This includes an 'absolute girder path' convenience column. The column '\_id' is the unique girder ID.  

* **items**: all items (slide). The column '\_id' is the unique girder ID, and is linked to the folders table by the 'folderId' column. 

* **annotation_docs**: Information about all the annotation documents (one document is a collection of elements like polygons, rectangles etc). The column 'annotation_girder_id' is the unique girder ID, and is linked to the 'items' table by the 'itemid' column. 

* **annotation_elements**: Information about the annotation elements (polygons, rectangles, points, etc). The column 'element_girder_id' is the unique girder ID, and is linked to the 'annotation_docs' table by the 'annotation_girder_id' column. 


**Syntax:**

It is a single function call, as in

.. code-block:: python

    import pandas as pd
    import sqlalchemy as db
    from histomicstk.annotations_and_masks.annotation_database_parser import (
        dump_annotations_locally, parse_annotations_to_local_tables)

    # recursively save annotations -- parse to sqlite
    dump_annotations_locally(
        gc, folderid=SAMPLE_FOLDER_ID, local=savepath,
        save_json=False, save_sqlite=True,
        callback=parse_annotations_to_local_tables,
        callback_kwargs={
            'save_csv': False,
            'save_sqlite': True,
        }
    )

    # connect to created database
    sql_engine = db.create_engine('sqlite:///%s/Concordance.sqlite' % savepath)
    dbcon = sql_engine.connect()

    # Now you can query results into a pandas table
    result = pd.read_sql_query("SELECT * FROM 'annotation_docs';", dbcon)


**Where to look:**

::

    |_histomicstk/
    |  |_annotations_and_masks/
    |     |_annotation_database_parser.py
    |     |_annotation_and_mask_utils.py -> parse_slide_annotations_into_tables()
    |     |_tests/
    |        |_annotation_database_parser_test.py
    |        |_annotation_and_mask_utils_test.py -> test_parse_slide_annotations_into_table()
    |_ docs/
       |_examples/
          |_annotation_database_parser.ipynb


**Screenshots:** 

.. image:: https://user-images.githubusercontent.com/22067552/72703220-09a62900-3b23-11ea-8968-709f938b1eb9.png
   :target: https://user-images.githubusercontent.com/22067552/72703220-09a62900-3b23-11ea-8968-709f938b1eb9.png
   :alt: image

.. image:: https://user-images.githubusercontent.com/22067552/72703277-29d5e800-3b23-11ea-80fe-86d82a4e86b3.png
   :target: https://user-images.githubusercontent.com/22067552/72703277-29d5e800-3b23-11ea-80fe-86d82a4e86b3.png
   :alt: image

.. image:: https://user-images.githubusercontent.com/22067552/72703918-001dc080-3b25-11ea-8ca2-6aa5454536db.png
   :target: https://user-images.githubusercontent.com/22067552/72703918-001dc080-3b25-11ea-8ca2-6aa5454536db.png
   :alt: image



.. automodule:: histomicstk.annotations_and_masks.annotation_database_parser
    :members:
    :undoc-members:
    :show-inheritance:
