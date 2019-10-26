# Tips of scalable annotation rendering for developers

**1. Some definitions and FAQs**

_- What is an Annotation?_ 

An annotation, aka annotation "document", is the single unit containing multiple elements (points/rectangles/lines/polygons/etc). When communicating with the HistomicsTK server using the `annotation` endpoint of the API, this is the indivisible unit used for sending and receiving data in a JSON-like format. The distinction between annotation documents, groups, and elements, is a bit suboptimal from a user standpoint, but is a useful abstraction for efficiency for developers (see discussion in #625 , #636). So even future changes to this model are likely to just impact display (i.e. abstract things away from the UI) but not the underlying principles. 

![image](https://user-images.githubusercontent.com/22067552/67130235-abd60e00-f1ce-11e9-82c7-50aeeb978c83.png)

_- What is an Element?_

An element, aka annotation "element". This is a single points/rectangles/lines/polygons/etc. Elements belonging to the same "document" are stored as a list as part of the "Elements" attribute of the document. 

_- Is there a limit to the number of loaded annotation documents?_

Yes! To avoid crashing the browser, there is a maximum number of annotation documents that are sampled for display. This number is currently set to 5,000, see #670 . This means that if your algorithm has saved more than 5,000 annotation documents, the end user will only be able to see 5,000 documents when they load the slide.

_- Is there a limit to the number of loaded annotation elements?_

Yes! If the number of elements in the documents is more than 2,000,000 then the smallest elements (as determined by bounding box diagonal) are displayed as circles, see #676 and #731 . Of course, if you have more than 5,000 documents in your slide then by definition some elements will be missing since only 5,000 documents are displayed. 

**2. Some efficiency considerations** - (Also see discussion at #636)

_- Which elements are more efficient to render?_

The basic types of elements include: point, rectangle, polygon, and line. The most efficient elements to render are, in order of increasing render efficiency: 

```
Filled polygon -> Unfilled polygon (no opacity) -> Rectangle -> Line -> Point
```

Obviously, the more vertices of the polygon the less is the rendering efficiency. 

_- What are the least performant operations for the end-user?_

1. Rendering thousands of filled polygons with numerous vertices (#592 ). 
2. Using the transparency slider - this re-renders the polygons, so does not perform well when there are thousands of filled polygons, see #672 
3. Right clicking and editing an element when there are thousands of RENDERED elements (see #533 ). For example, say the slide is associated with 500 annotation documents, each of which contains 2,000 annotation elements. If the user clicks the 'eye' icon to render ALL 500 x 2000 = 100,000 elements, then right clicking on any single element to edit will be very slow. If the user just chooses to render a couple of documents (i.e. only 1,000 elements) at a time, then there is no performance issue.

**3. Long story short, what should I do?**

_- "Spread out" your annotation elements over many annotation documents_

Of course, keeping in mind that you do not exceed the maximum limit of 5,000 documents. This makes sure the server sends and receives 'bite-sized' (in a non-computer-science sense) packets of data, which is good for everyone. Here's why:
1. It is good for pushing data using the `POST` or editing data using the `PUT` endpoints using the API without getting a request time out. 
2. It makes sure the user can edit the individual elements without problems. Note that right clicking an editing an element using the UI uses the `PUT` endpoint to edit the `elements` attribute of the annotation document to which the to-be-edited element belongs. So even if the user wants to edit one single annotation element, the client sends a `PUT` request containing the entire new, edited annotation document. Is this optimal behavior? It could probably be improved, but for now, keep this in mind. (Also see discussion in #719).
3. It allows the user to display groups of elements at a time if the user really desires highly performant rendering, see #592 . 

_- Consider which elements to group together into the same document_

This depends on the use case. For example, you may decide to group all elements from the same region-of-interest (or geographic location) into the same document so that the user can view all annotations in the same locality with one click on the "eye" icon. If there is no natural "region of interest" in your project and you just have tens of thousands of objects spread throughout the slide, consider grouping them by geographic proximity. 

Alternatively, you may decide to have elements of the same group (eg tumor) to be in the same document. This relates to #698 and #592  . Again, even if you elect to use this option, make sure to spread things over many documents if possible. 

_- Minimize number of filled polygons_

This goes without saying. Any polygon fill is rendered on the GPU and is re-rendered with the interactive mode, transparency slider, right click, multi-select, lasso select, etc. 

_- Minimize vertices per polygon_

If your polygons are displaying algorithmic outputs, and they were originally part of a mask at, say. 40x magnification, consider resizing the source mask to a smaller magnification (say, 15x) before extracting the polygonal coordinates. If you don't expect the user to need a level of detail beyond 15x, only use enough vertices to be accurate at 15x. 

_- Make sure the end-user knows about efficiency bottlenecks_

If your end user (eg pathology resident) needs to edit elements in a slide with hundreds of thousands of elements, make sure to let them know which things to do and not to do. 
