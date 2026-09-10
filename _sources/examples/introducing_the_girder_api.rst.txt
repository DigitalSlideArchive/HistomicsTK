Introducing the Girder API
==========================

A RESTful API allows developers to programmatically interact with a DSA server
to manage data, user accounts, annotations, and their permissions. These
functions are part of what makes DSA a powerful platform for generating
annotations through distributed multi-user studies. In this example we discuss
the basic elements of the DSA database, describe the structure of annotations,
and illustrate a variety of methods for making DSA API calls.

The girder database
--------------------

DSA is built using `girder`_, an open source web-based data management
platform developed by `Kitware`_. The DSA server uses a `MongoDB`_ database to
organize images, image metadata, user accounts, and annotations. The DSA
user interface allows basic interaction with this data through a web
browser and can perform administrative tasks. Tasks requiring programmatic
access can use the `girder python client`_, a `RESTful API`_ client that provides
convenient endpoints for interacting with and managing data hosted on a DSA
instance. The girder database contains the following elements that can be
manipulated using the API:

- Users and Groups
- Collections and Folders
- Items and Files
- Annotations
- Assetstores

The `girder user guide`_ describes these concepts in detail along with API call
formats. User-level and group-level permissions provide very granular control
over access to data and annotations as described
`here <https://girder.readthedocs.io/en/stable/user-guide.html#permissions>`_.
See *“Procedure for managing a typical annotation project”* for more details on
the permissions model and its role in multi-user annotation studies.

What is an annotation?
------------------------

The annotation schema used by HistomicsUI for image markup is provided by
the `large_image`_ library and is `described here <https://github.com/girder/large_image/blob/master/girder_annotation/docs/annotations.md>`_.

An annotation document, or simply annotation, contains a set of elements (points, rectangles, lines, polygons, etc.) and serves to organize these to simplify analysis or to enhance rendering performance when large numbers of elements are present. When using the API an annotation document is the indivisible unit used for sending and receiving annotation data. Individual elements have style attributes like line and fill color that define their appearance, and these attributes can be uniform or varied within the same document. The way that users experience conceptual grouping of elements is through groups that are each linked to a standard style. Groups can cut across documents and represent concepts like different types of cells or tissue regions. Groups and their styles can be created through the HistomicsUI interface, and a file defining group names and their styles can be downloaded in a JSON format and distributed to users to help standardize naming and appearances in multi-user annotation studies.

.. image:: https://user-images.githubusercontent.com/22067552/67130235-abd60e00-f1ce-11e9-82c7-50aeeb978c83.png

The distinction between annotation documents, elements and groups, is a bit confusing from a user standpoint, but is a useful abstraction for efficiency for developers.

Using the girder API
----------------------

As a developer, most of your interaction with DSA will occur through the API
using the `python client library and/or command-line interface <https://girder.readthedocs.io/en/stable/python-client.html#>`_.
The client simplifies the formulation of API calls and allows you to authenticate using your username and password or an API key. In many situations it is more secure to use API keys because:

- You can create multiple keys with different permissions for different roles. This gives you the flexibility to provide the key to multiple people or to share the key in demonstrations/videos etc.
- You can disable or delete a key anytime.
- Some interactive shells do not hide passwords in interactive mode. The password could be visible to others on your screen and could be saved in the shell history.

API keys can be created and managed through the DSA user interface or API. Following authentication you can use the REST methods to query or manipulate data hosted on the DSA instance using:

- **GET**: Get (download) resource (item, folder information, slide region, annotation, etc).
- **PUT**: Modify an existing resource.
- **POST**: Post a new resource to the DSA server.
- **DELETE**: Delete a resource from the DSA server.

These commands are used in conjunction with API endpoints that implement common
tasks. For example, a GET method can be used with the annotation endpoint to
download the annotation from the server. The full list of endpoints is
available on the interactive web API docs which are linked on the landing
page of your DSA server. `Here is the API documentation <https://demo.kitware.com/histomicstk/api/v1>`_ from our demo DSA
instance. There is more than one way to use the API, and there are a number
of higher level methods we have to make your life easier. Here are some options:

- Manually through the web API hosted on your DSA instance

    The web API hosted on your DSA instance provides a simple interface where you can manually make API calls to perform simple jobs or to test API calls. Note that this is not a sandbox and calls made through the interface have lasting impact.

- Programmatically using the girder client with get/put/post/delete methods

    There are two ways to use the girder client with get/put/post/delete methods.

    The first is to formulate an HTTP request string and pass is to the method.

    .. code-block:: python

        from histomicstk.utils.girder_convenience_utils import connect_to_api

        gc = connect_to_api(apiurl)

        # fetch rgb region
        getStr = \
            "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" \
            % (slide_id, left, right, top, bottom)
        resp = gc.get(getStr, jsonResp=False)


    For sending large amounts of data, it is preferable to use the data parameter to pass parameters as a dictionary

    .. code-block:: python

        # update annotation permissions
        resp = gc.put(
            '/annotation/%s/access' % annotation_id,
            data={'access': json.dumps(new_access_permissions_dict)},
        )


- Programmatically using girder client utility methods

    The girder client provides built-in methods that automate common tasks
    for dealing with collections, users, items, and annotations.
    These methods avoid the need to formulate complex API call request
    strings. Documentation of these methods is
    `available here <https://girder.readthedocs.io/en/stable/python-client.html#the-python-client-library>`_.

- Using HistomicsTK functions with the girder client

    HistomicsTK contains methods that use the girder client to perform
    operations like `applying image analysis functions <workflows.ipynb>`_ to a set of remotely
    hosted slides, or for `handling annotations <procedure_for_typical_annotation_project.rst>`_.

.. _RESTful API: https://restfulapi.net/
.. _girder: https://girder.readthedocs.io/en/stable/index.html
.. _Kitware: https://www.kitware.com/
.. _MongoDB: https://www.mongodb.com/
.. _girder python client: https://girder.readthedocs.io/en/stable/api-docs.html
.. _girder user guide: https://girder.readthedocs.io/en/stable/user-guide.html
.. _large_image: https://github.com/girder/large_image
