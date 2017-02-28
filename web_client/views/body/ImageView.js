/* global geo */
import _ from 'underscore';

import { restRequest } from 'girder/rest';
import ItemModel from 'girder/models/ItemModel';
import FileModel from 'girder/models/FileModel';
import GeojsViewer from 'girder_plugins/large_image/views/imageViewerWidget/geojs';
import SlicerPanelGroup from 'girder_plugins/slicer_cli_web/views/PanelGroup';

import AnnotationSelector from '../../panels/AnnotationSelector';
import router from '../../router';
import events from '../../events';
import View from '../View';

import imageTemplate from '../../templates/body/image.pug';
import '../../stylesheets/body/image.styl';

var ImageView = View.extend({
    events: {},
    initialize(settings) {
        this.viewerWidget = null;
        this._openId = null;
        if (!this.model) {
            this.model = new ItemModel();
        }
        this.listenTo(this.model, 'g:fetched', this.render);
        this.listenTo(events, 'h:analysis', this._setImageInput);
        events.trigger('h:imageOpened', null);
        this.listenTo(events, 'query:image', this.openImage);
        this.controlPanel = new SlicerPanelGroup({
            parentView: this
        });
        this.render();
    },
    render() {
        if (this.model.id === this._openId) {
            this.controlPanel.setElement('.h-control-panel-container').render();
            return;
        }
        this.$el.html(imageTemplate());
        if (this.model.id) {
            this._openId = this.model.id;
            this.viewerWidget = new GeojsViewer({
                parentView: this,
                el: this.$('.h-image-view-container'),
                itemId: this.model.id
            });
            this.viewerWidget.on('g:imageRendered', () => {
                events.trigger('h:imageOpened', this.model);
                // store a reference to the underlying viewer
                this.viewer = this.viewerWidget.viewer;

                // set the viewer bounds on first load
                this.setImageBounds();

                // update the query string on pan events
                if (this.viewer) {
                    this.viewer.geoOn(geo.event.pan, () => {
                        this.setBoundsQuery();
                    });
                }
            });
            this.annotationSelector = new AnnotationSelector({
                parentView: this,
                model: this.model
            });
            this.annotationSelector.setElement('.h-annotation-selector').render();
        }
        this.controlPanel.setElement('.h-control-panel-container').render();
    },
    destroy() {
        if (this.viewerWidget) {
            this.viewerWidget.destroy();
        }
        this.viewerWidget = null;
        events.trigger('h:imageOpened', null);
        return View.prototype.destroy.apply(this, arguments);
    },
    openImage(id) {
        if (id) {
            this.model.set({_id: id}).fetch().then(() => {
                this._setImageInput();
            });
        } else {
            this.model.set({_id: null});
            this.render();
        }
    },
    /**
     * Set any input image parameters to the currently open image.
     * The jobs endpoints expect file id's rather than item id's,
     * so we have to choose an appropriate file id for a number of
     * scenarios.
     *
     *  * A normal item: Pick the first file id.  Here we have
     *    to make another rest call to get the files contained
     *    in the item.
     *
     *  * A large image item: choose originalId over fileId
     *    because slicer endpoints can't yet handle tiled image
     *    formats.
     *
     *  After getting the file id we have to make another rest
     *  call to fetch the full file model from the server.  Once
     *  this is complete, set the widget value.
     */
    _setImageInput() {
        if (!this.model.id) {
            return;
        }

        // helper functions passed through promises
        var getItemFile = (itemId) => {
            return restRequest({
                path: 'item/' + itemId + '/files',
                data: {
                    limit: 1,
                    offset: 0
                }
            }).then((files) => {
                if (!files.length) {
                    throw new Error('Item does not contain a file.');
                }
                return new FileModel(files[0]);
            });
        };

        var getFileModel = (fileId) => {
            return restRequest({
                path: 'file/' + fileId
            }).then((file) => {
                return new FileModel(file);
            });
        };
        var largeImage = this.model.get('largeImage');
        var promise;

        if (largeImage) {
            // Until slicer jobs can handle tiled input formats use
            // the original file if available.
            promise = getFileModel(largeImage.originalId || largeImage.fileId);
        } else {
            promise = getItemFile(this.model.id);
        }

        return promise.then((file) => {
            _.each(this.controlPanel.models(), (model) => {
                if (model.get('type') === 'image') {
                    model.set('value', file, {trigger: true});
                }
            });
        });
    },

    /**
     * Set the view (image bounds) of the current image as a
     * query string parameter.
     */
    setBoundsQuery() {
        var bounds, left, right, top, bottom, rotation;
        if (this.viewer) {
            bounds = this.viewer.bounds();
            rotation = (this.viewer.rotation() * 180 / Math.PI).toFixed();
            left = bounds.left.toFixed();
            right = bounds.right.toFixed();
            top = bounds.top.toFixed();
            bottom = bounds.bottom.toFixed();
            router.setQuery('bounds', [
                left, top, right, bottom, rotation
            ].join(','));
        }
    },

    /**
     * Get the view from the query string and set it on the image.
     */
    setImageBounds() {
        var bounds = router.getQuery('bounds');
        if (!bounds || !this.viewer) {
            return;
        }
        bounds = bounds.split(',');
        this.viewer.bounds({
            left: parseFloat(bounds[0]),
            top: parseFloat(bounds[1]),
            right: parseFloat(bounds[2]),
            bottom: parseFloat(bounds[3])
        });
        var rotation = parseFloat(bounds[4]) || 0;
        this.viewer.rotation(rotation * Math.PI / 180);
    }
});

export default ImageView;
