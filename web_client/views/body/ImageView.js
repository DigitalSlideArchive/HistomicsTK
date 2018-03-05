/* global geo */
import _ from 'underscore';

import { restRequest } from 'girder/rest';
import { getCurrentUser } from 'girder/auth';
import ItemModel from 'girder/models/ItemModel';
import FileModel from 'girder/models/FileModel';
import FolderCollection from 'girder/collections/FolderCollection';
import GeojsViewer from 'girder_plugins/large_image/views/imageViewerWidget/geojs';
import SlicerPanelGroup from 'girder_plugins/slicer_cli_web/views/PanelGroup';
import AnnotationModel from 'girder_plugins/large_image/models/AnnotationModel';
import AnnotationCollection from 'girder_plugins/large_image/collections/AnnotationCollection';

import AnnotationPopover from '../popover/AnnotationPopover';
import AnnotationSelector from '../../panels/AnnotationSelector';
import ZoomWidget from '../../panels/ZoomWidget';
import DrawWidget from '../../panels/DrawWidget';
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
        this._displayedRegion = null;

        if (!this.model) {
            this.model = new ItemModel();
        }
        this.listenTo(this.model, 'g:fetched', this.render);
        this.listenTo(events, 'h:analysis', this._setImageInput);
        this.listenTo(events, 'h:analysis', this._setDefaultFileOutputs);
        this.listenTo(events, 'h:analysis', this._resetRegion);
        events.trigger('h:imageOpened', null);
        this.listenTo(events, 'query:image', this.openImage);
        this.annotations = new AnnotationCollection();

        this.controlPanel = new SlicerPanelGroup({
            parentView: this
        });
        this.zoomWidget = new ZoomWidget({
            parentView: this
        });
        this.annotationSelector = new AnnotationSelector({
            parentView: this,
            collection: this.annotations,
            image: this.model
        });
        this.popover = new AnnotationPopover({
            parentView: this
        });

        this.listenTo(events, 'h:select-region', this.showRegion);
        this.listenTo(this.annotationSelector.collection, 'add update change:displayed', this.toggleAnnotation);
        this.listenTo(this.annotationSelector, 'h:toggleLabels', this.toggleLabels);
        this.listenTo(this.annotationSelector, 'h:editAnnotation', this._editAnnotation);
        this.listenTo(this.annotationSelector, 'h:deleteAnnotation', this._deleteAnnotation);
        this.listenTo(this, 'h:highlightAnnotation', this._highlightAnnotation);

        this.listenTo(events, 's:widgetChanged:region', this.widgetRegion);
        this.listenTo(events, 'g:login g:logout.success g:logout.error', () => {
            this._openId = null;
        });
        this.render();
    },
    render() {
        // Ensure annotations are removed from the popover widget on rerender.
        // This can happen when opening a new image while an annotation is
        // being hovered.
        this.mouseResetAnnotation();
        this._removeDrawWidget();

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
                itemId: this.model.id,
                hoverEvents: true,
                highlightFeatureSizeLimit: 1000
            });
            this.trigger('h:viewerWidgetCreated', this.viewerWidget);

            // handle annotation mouse events
            this.listenTo(this.viewerWidget, 'g:mouseOverAnnotation', this.mouseOverAnnotation);
            this.listenTo(this.viewerWidget, 'g:mouseOutAnnotation', this.mouseOutAnnotation);
            this.listenTo(this.viewerWidget, 'g:mouseOnAnnotation', this.mouseOnAnnotation);
            this.listenTo(this.viewerWidget, 'g:mouseOffAnnotation', this.mouseOffAnnotation);
            this.listenTo(this.viewerWidget, 'g:mouseClickAnnotation', this.mouseClickAnnotation);
            this.listenTo(this.viewerWidget, 'g:mouseResetAnnotation', this.mouseResetAnnotation);

            this.viewerWidget.on('g:imageRendered', () => {
                events.trigger('h:imageOpened', this.model);
                // store a reference to the underlying viewer
                this.viewer = this.viewerWidget.viewer;

                // set the viewer bounds on first load
                this.setImageBounds();

                // also set the query string
                this.setBoundsQuery();

                if (this.viewer) {
                    // update the query string on pan events
                    this.viewer.geoOn(geo.event.pan, () => {
                        this.setBoundsQuery();
                    });

                    // update the coordinate display on mouse move
                    this.viewer.geoOn(geo.event.mousemove, (evt) => {
                        this.showCoordinates(evt);
                    });

                    // remove the hidden class from the coordinates display
                    this.$('.h-image-coordinates-container').removeClass('hidden');

                    // show the right side control container
                    this.$('#h-annotation-selector-container').removeClass('hidden');

                    this.zoomWidget
                        .setViewer(this.viewerWidget)
                        .setElement('.h-zoom-widget').render();

                    this.annotationSelector
                        .setViewer(this.viewerWidget)
                        .setElement('.h-annotation-selector').render();

                    if (this.drawWidget) {
                        this.$('.h-draw-widget').removeClass('hidden');
                        this.drawWidget
                            .setViewer(this.viewerWidget)
                            .setElement('.h-draw-widget').render();
                    }
                }
            });
            this.annotationSelector.setItem(this.model);

            this.annotationSelector
                .setViewer(null)
                .setElement('.h-annotation-selector').render();

            if (this.drawWidget) {
                this.$('.h-draw-widget').removeClass('hidden');
                this.drawWidget
                    .setViewer(null)
                    .setElement('.h-draw-widget').render();
            }
        }
        this.controlPanel.setElement('.h-control-panel-container').render();
        this.popover.setElement('#h-annotation-popover-container').render();
        return this;
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
                return null;
            });
        } else {
            this.model.set({_id: null});
            this.render();
            this._openId = null;
            events.trigger('h:imageOpened', null);
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
     *  * A large image item: choose fileId over originalId.
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
                url: 'item/' + itemId + '/files',
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

        var getTilesDef = (itemId) => {
            return restRequest({
                url: 'item/' + itemId + '/tiles'
            }).then((tiles) => {
                this.zoomWidget.setMaxMagnification(tiles.magnification || 20);
                return null;
            });
        };

        var getFileModel = (fileId) => {
            return restRequest({
                url: 'file/' + fileId
            }).then((file) => {
                return new FileModel(file);
            });
        };
        var largeImage = this.model.get('largeImage');
        var promise;

        if (largeImage) {
            // Prefer the fileId, expecting that jobs can handle tiled input
            promise = $.when(
                getTilesDef(this.model.id),
                getFileModel(largeImage.fileId || largeImage.originalId)
            ).then((a, b) => b); // resolve with the file model
        } else {
            promise = getItemFile(this.model.id);
        }

        return promise.then((file) => {
            _.each(this.controlPanel.models(), (model) => {
                if (model.get('type') === 'image') {
                    model.set('value', file, {trigger: true});
                }
            });
            return null;
        });
    },

    _getDefaultOutputFolder() {
        const user = getCurrentUser();
        if (!user) {
            return;
        }
        const userFolders = new FolderCollection();
        return userFolders.fetch({
            parentId: user.id,
            parentType: 'user',
            name: 'Private',
            limit: 1
        }).then(() => {
            if (userFolders.isEmpty()) {
                throw new Error('Could not find the user\'s private folder when setting defaults');
            }
            return userFolders.at(0);
        });
    },

    _setDefaultFileOutputs() {
        return this._getDefaultOutputFolder().done((folder) => {
            _.each(
                this.controlPanel.models().filter((model) => model.get('type') === 'new-file'),
                (model) => {
                    var analysis = _.last(router.getQuery('analysis').split('/'));
                    var extension = (model.get('extensions') || '').split('|')[0];
                    var name = `${analysis}-${model.id}${extension}`;
                    model.set({
                        path: [folder.get('name'), name],
                        parent: folder,
                        value: new ItemModel({
                            name,
                            folderId: folder.id
                        })
                    });
                }
            );
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
            ].join(','), {replace: true});
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
    },

    toggleAnnotation(annotation) {
        if (!this.viewerWidget) {
            // We may need a way to queue annotation draws while viewer
            // initializes, but for now ignore them.
            return;
        }

        if (annotation.get('displayed')) {
            annotation.set('loading', true);
            annotation.fetch().then(() => {
                this.viewerWidget.drawAnnotation(annotation);
                return null;
            }).always(() => {
                annotation.unset('loading');
            });
        } else {
            this.viewerWidget.removeAnnotation(annotation);
        }
    },

    _redrawAnnotation(annotation) {
        if (!this.viewerWidget || !annotation.get('displayed')) {
            // We may need a way to queue annotation draws while viewer
            // initializes, but for now ignore them.
            return;
        }
        this.viewerWidget.drawAnnotation(annotation);
    },

    _highlightAnnotation(annotation, element) {
        this.viewerWidget.highlightAnnotation(annotation, element);
    },

    widgetRegion(model) {
        var value = model.get('value');
        this._displayedRegion = value.slice();
        this.showRegion({
            left: parseFloat(value[0]),
            right: parseFloat(value[0]) + parseFloat(value[2]),
            top: parseFloat(value[1]),
            bottom: parseFloat(value[1]) + parseFloat(value[3])
        });
    },

    _resetRegion() {
        var hasRegionParameter;
        if (!this._displayedRegion) {
            return;
        }
        _.each(
            this.controlPanel.models().filter((model) => model.get('type') === 'region'),
            (model) => {
                model.set('value', this._displayedRegion);
                hasRegionParameter = true;
            }
        );
        if (!hasRegionParameter) {
            this._displayedRegion = null;
            this.showRegion(null);
        }
    },

    showRegion(region) {
        if (this.viewerWidget) {
            this.viewerWidget.removeAnnotation(
                new AnnotationModel({_id: 'region-selection'})
            );
        }

        if (!region) {
            return;
        }

        var center = [
            (region.left + region.right) / 2,
            (region.top + region.bottom) / 2,
            0
        ];
        var width = region.right - region.left;
        var height = region.bottom - region.top;
        var fillColor = 'rgba(255,255,255,0)';
        var lineColor = 'rgba(0,0,0,1)';
        var lineWidth = 2;
        var rotation = 0;
        var annotation = new AnnotationModel({
            _id: 'region-selection',
            name: 'Region',
            annotation: {
                elements: [{
                    type: 'rectangle',
                    center,
                    width,
                    height,
                    rotation,
                    fillColor,
                    lineColor,
                    lineWidth
                }]
            }
        });
        this.viewerWidget.drawAnnotation(annotation, {fetch: false});
    },

    showCoordinates(evt) {
        if (this.viewer) {
            var pt = evt.geo;
            this.$('.h-image-coordinates').text(
                pt.x.toFixed() + ', ' + pt.y.toFixed()
            );
        }
    },

    mouseOnAnnotation(element, annotationId) {
        if (annotationId === 'region-selection') {
            return;
        }
        const annotation = this.annotations.get(annotationId);
        const elementModel = annotation.elements().get(element.id);
        annotation.set('highlight', true);
        if (this.drawWidget) {
            this.drawWidget.trigger('h:mouseon', elementModel);
        }
    },

    mouseOffAnnotation(element, annotationId) {
        if (annotationId === 'region-selection') {
            return;
        }
        const annotation = this.annotations.get(annotationId);
        const elementModel = annotation.elements().get(element.id);
        annotation.unset('highlight');
        if (this.drawWidget) {
            this.drawWidget.trigger('h:mouseoff', elementModel);
        }
    },

    mouseOverAnnotation(element, annotationId) {
        if (annotationId === 'region-selection') {
            return;
        }
        element.annotation = this.annotations.get(annotationId);
        if (element.annotation) {
            this.popover.collection.add(element);
        }
    },

    mouseOutAnnotation(element, annotationId) {
        if (annotationId === 'region-selection') {
            return;
        }
        element.annotation = this.annotations.get(annotationId);
        if (element.annotation) {
            this.popover.collection.remove(element);
        }
    },

    mouseResetAnnotation() {
        this.popover.collection.reset();
    },

    mouseClickAnnotation(/* element, annotationId */) {
    },

    toggleLabels(options) {
        this.popover.toggle(options.show);
    },

    _removeDrawWidget() {
        if (this.drawWidget) {
            this._lastDrawingType = this.drawWidget.drawingType();
            this.drawWidget.cancelDrawMode();
            this.stopListening(this.drawWidget);
            this.drawWidget.remove();
            this.drawWidget = null;
            $('<div/>').addClass('h-draw-widget s-panel hidden')
                .appendTo(this.$('#h-annotation-selector-container'));
        }
    },

    _editAnnotation(model) {
        this.activeAnnotation = model;
        this._removeDrawWidget();
        if (model) {
            this.drawWidget = new DrawWidget({
                parentView: this,
                image: this.model,
                annotation: this.activeAnnotation,
                drawingType: this._lastDrawingType,
                el: this.$('.h-draw-widget'),
                viewer: this.viewerWidget
            }).render();
            this.listenTo(this.drawWidget, 'h:redraw', this._redrawAnnotation);
            this.$('.h-draw-widget').removeClass('hidden');
        }
    },

    _deleteAnnotation(model) {
        if (this.activeAnnotation && this.activeAnnotation.id === model.id) {
            this._removeDrawWidget();
        }
    }
});

export default ImageView;
