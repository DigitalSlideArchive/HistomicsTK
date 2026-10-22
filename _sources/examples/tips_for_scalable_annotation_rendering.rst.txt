Tips for scaling annotation rendering
======================================

Image analysis algorithms and humans can generate large numbers of annotations.
HistomicsUI is implemented for scalable rendering of annotations, and if you
follow some basic principles you can avoid performance problems in rendering
large numbers of elements and vertices.

**The maximum number of annotation documents that HistomicsUI can display is 5,000**

HistomicsUK imposes this limit to avoid crashing the browser. If a slide contains more than 5,000 documents then HistomicsUI will sample 5,000 of these to display.

**The maximum number of elements that HistomicsUI can display is 2,000,000.**

This limit applies to the combined elements in all documents in the slide. If more than 2,000,000 elements are present then the smallest elements are displayed as circles (as determined by bounding box diagonal).

**Which element types are more efficient to render?**

The key types of elements include: point, rectangle, polygon, and line. The most efficient elements to render are, in order of increasing render efficiency:

    ``Filled polygon → Unfilled polygon (no opacity) → Rectangle → Line → Point``

Obviously, the more vertices of the polygon the less is the rendering efficiency.

**What end-user operations can lead to performance issues?**

- Rendering thousands of filled polygons or complex polygons with many vertices.
- Using the transparency slider. Each step of the slider causes the polygons to be re-rendered which can be slow when thousands of filled polygons are present.
- Right click editing of an element when thousands of elements are being rendered. For example, say the slide is associated with 500 annotation documents, each of which contains 2,000 annotation elements. If the user clicks the ‘eye’ icon to render all 500 x 2000 = 100,000 elements, then right clicking on any single element to edit will be very slow. If the user just chooses to render a couple of documents (i.e. only 1,000 elements) at a time, then there is no performance issue.

**How can I improve rendering performance with large numbers of elements?**

- *Distribute your elements over several annotation documents*

    This helps improve server communication in several way

    - Reducing likelihood of request timeouts when pushing annotation data using POST or editing data using PUT.

    - Minimizing communication volume when editing annotations via HistomicsUI. Users can right click edit an individual element which triggers a PUT endpoint to send the updated document back to the DSA instance. In the future we will improve how edits are communicated but for now keep this in mind.

    - Allows the user to selectively display groups of elements and maintain rendering performance when the slide contains very large numbers of elements.

- *Consider which elements should be grouped into the same documents*

    Organizing elements by region can help users control rendering more easily.
    For example, given multiple pieces of tissue in a slide, you could assign a
    document to each tissue piece, allowing users to control the rendering on
    each piece independently. Alternatively, you may assign elements of
    conceptual styles (e.g. tumor) to the same document for simplicity. If you
    elect to do this, you may need multiple documents per group to maintain performance.

- *Minimize the use of polygon fills*

    Any polygon fills are rendered on the GPU and is re-rendered with the
    interactive mode, transparency slider, right click, multi-select, lasso select, etc.

- *Minimize vertices per polygon*

    If your polygons represent algorithmic outputs then consider how you can
    minimize the number of vertices. If they were originally generated from a
    mask image at high magnification 20x or above, consider resizing the source
    mask to a smaller magnification before extracting . Post processing of
    polygonal coordinates can also be done to remove redundant collinear vertices
    or to simplify vertices.

- *Make your end-users aware of efficiency bottlenecks*

    If your end user needs to edit elements in a slide with hundreds of
    thousands of elements, make sure to let them know which things to do and
    not to do. The user should not be viewing all annotations simultaneously
    unless the number is sufficiently small or there’s a specific reason to do
    so. They should be made aware that right clicking and editing annotations
    and using the transparency slider are performance bottlenecks when there are
    thousands of rendered annotations.
