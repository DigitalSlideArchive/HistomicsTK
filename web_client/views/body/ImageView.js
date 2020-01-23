/* global geo */
import _ from 'underscore';

import { restRequest } from 'girder/rest';
import { getCurrentUser } from 'girder/auth';
import { AccessType } from 'girder/constants';
import ItemModel from 'girder/models/ItemModel';
import FileModel from 'girder/models/FileModel';
import FolderCollection from 'girder/collections/FolderCollection';
import GeojsViewer from 'girder_plugins/large_image/views/imageViewerWidget/geojs';
import SlicerPanelGroup from 'girder_plugins/slicer_cli_web/views/PanelGroup';
import AnnotationModel from 'girder_plugins/large_image/models/AnnotationModel';
import AnnotationCollection from 'girder_plugins/large_image/collections/AnnotationCollection';

import AnnotationContextMenu from '../popover/AnnotationContextMenu';
import AnnotationPopover from '../popover/AnnotationPopover';
import AnnotationSelector from '../../panels/AnnotationSelector';
import ZoomWidget from '../../panels/ZoomWidget';
import DrawWidget from '../../panels/DrawWidget';
import editElement from '../../dialogs/editElement';
import router from '../../router';
import events from '../../events';
import View from '../View';

import imageTemplate from '../../templates/body/image.pug';
import '../../stylesheets/body/image.styl';

var ImageView = View.extend({
    events: {
        'keydown .h-image-body': '_onKeyDown',
        'keydown .geojs-map': '_handleKeyDown',
        'click .h-control-panel-container .s-close-panel-group': '_closeAnalysis',
        'mousemove .geojs-map': '_trackMousePosition'
    },
    initialize(settings) {
        this.viewerWidget = null;
        this._mouseClickQueue = [];
        this._openId = null;
        this._displayedRegion = null;
        this._currentMousePosition = null;
        this._selectElementsByRegionCanceled = false;
        this.selectedAnnotation = new AnnotationModel({ _id: 'selected' });
        this.selectedElements = this.selectedAnnotation.elements();

        // Allow zooming this many powers of 2 more than native pixel resolution
        this._increaseZoom2x = 1;

        if (!this.model) {
            this.model = new ItemModel();
        }
        this.listenTo(this.model, 'g:fetched', this.render);
        this.listenTo(events, 'h:analysis:rendered', this._setImageInput);
        this.listenTo(events, 'h:analysis:rendered', this._setDefaultFileOutputs);
        this.listenTo(events, 'h:analysis:rendered', this._resetRegion);
        this.listenTo(this.selectedElements, 'add remove reset', this._redrawSelection);
        events.trigger('h:imageOpened', null);
        this.listenTo(events, 'query:image', this.openImage);
        this.annotations = new AnnotationCollection();

        this.controlPanel = new SlicerPanelGroup({
            parentView: this,
            closeButton: true
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
        this.contextMenu = new AnnotationContextMenu({
            parentView: this,
            collection: this.selectedElements
        });
        this.listenTo(this, 'h:styleGroupsEdited', () => {
            this.contextMenu.refetchStyles();
        });

        this.listenTo(this.annotationSelector, 'h:groupCount', (obj) => {
            this.contextMenu.setGroupCount(obj);
        });
        this.listenTo(events, 'h:submit', (data) => {
            this.$('.s-jobs-panel .s-panel-controls .icon-down-open').click();
            events.trigger('g:alert', { type: 'success', text: 'Analysis job submitted.' });
        });
        this.listenTo(events, 'h:select-region', this.showRegion);
        this.listenTo(this.annotationSelector.collection, 'add update change:displayed', this.toggleAnnotation);
        this.listenTo(this.annotationSelector, 'h:toggleLabels', this.toggleLabels);
        this.listenTo(this.annotationSelector, 'h:toggleInteractiveMode', this._toggleInteractiveMode);
        this.listenTo(this.annotationSelector, 'h:editAnnotation', this._editAnnotation);
        this.listenTo(this.annotationSelector, 'h:deleteAnnotation', this._deleteAnnotation);
        this.listenTo(this.annotationSelector, 'h:annotationOpacity', this._setAnnotationOpacity);
        this.listenTo(this.annotationSelector, 'h:annotationFillOpacity', this._setAnnotationFillOpacity);
        this.listenTo(this.annotationSelector, 'h:redraw', this._redrawAnnotation);
        this.listenTo(this, 'h:highlightAnnotation', this._highlightAnnotationForInteractiveMode);
        this.listenTo(this, 'h:selectElementsByRegion', this._selectElementsByRegion);
        this.listenTo(this, 'h:selectElementsByRegionCancel', this._selectElementsByRegionCancel);
        this.listenTo(this.contextMenu, 'h:edit', this._editElement);
        this.listenTo(this.contextMenu, 'h:redraw', this._redrawAnnotation);
        this.listenTo(this.contextMenu, 'h:close', this._closeContextMenu);
        this.listenTo(this.selectedElements, 'h:save', this._saveSelection);
        this.listenTo(this.selectedElements, 'h:remove', this._removeSelection);

        this.listenTo(events, 's:widgetChanged:region', this.widgetRegion);
        this.listenTo(events, 'g:login g:logout.success g:logout.error', () => {
            this._openId = null;
            this.model.set({ _id: null });
        });
        $(document).on('mousedown.h-image-view', (evt) => {
            // let the context menu close itself
            if ($(evt.target).parents('#h-annotation-context-menu').length) {
                return;
            }
            this._closeContextMenu();
        });
        $(document).on('keydown.h-image-view', (evt) => {
            if (evt.keyCode === 27) {
                this._closeContextMenu();
            }
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
        this.contextMenu.setElement(this.$('#h-annotation-context-menu')).render();

        if (this.model.id) {
            this._openId = this.model.id;
            if (this.viewerWidget) {
                this.viewerWidget.destroy();
            }
            this.viewerWidget = new GeojsViewer({
                parentView: this,
                el: this.$('.h-image-view-container'),
                itemId: this.model.id,
                hoverEvents: true,
                // it is very confusing if this value is smaller than the
                // AnnotationSelector MAX_ELEMENTS_LIST_LENGTH
                highlightFeatureSizeLimit: 5000,
                scale: { position: { bottom: 20, right: 10 } }
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

                this.imageWidth = this.viewer.maxBounds().right;
                this.imageHeight = this.viewer.maxBounds().bottom;
                // allow panning off the image slightly
                var extraPanWidth = 0.1, extraPanHeight = 0;
                this.viewer.maxBounds({
                    left: -this.imageWidth * extraPanWidth,
                    right: this.imageWidth * (1 + extraPanWidth),
                    top: -this.imageHeight * extraPanHeight,
                    bottom: this.imageHeight * (1 + extraPanHeight)
                });

                // set the viewer bounds on first load
                this.setImageBounds();

                // also set the query string
                this.setBoundsQuery();

                if (this.viewer) {
                    this.viewer.zoomRange({ max: this.viewer.zoomRange().max + this._increaseZoom2x });

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
        $(document).off('.h-image-view');
        return View.prototype.destroy.apply(this, arguments);
    },
    openImage(id) {
        /* eslint-disable backbone/no-silent */
        this.model.clear({silent: true});
        delete this.model.parent;
        if (id) {
            this.model.set({ _id: id }).fetch().then(() => {
                this._setImageInput();
                return null;
            });
        } else {
            this.model.set({ _id: null });
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
                this.zoomWidget.setMaxMagnification(tiles.magnification || 20, this._increaseZoom2x);
                this.zoomWidget.render();
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
                    model.set('value', file, { trigger: true });
                }
            });
            return null;
        });
    },

    _getDefaultOutputFolder() {
        const user = getCurrentUser();
        if (!user) {
            return $.Deferred().resolve().promise();
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
            if (folder) {
                _.each(
                    this.controlPanel.models().filter((model) => model.get('type') === 'new-file'),
                    (model) => {
                        var analysis = _.last(router.getQuery('analysis').split('/'));
                        var extension = (model.get('extensions') || '').split('|')[0];
                        var name = `${analysis}-${model.id}${extension}`;
                        if (model.get('required') !== false) {
                            model.set({
                                path: [folder.get('name'), name],
                                parent: folder,
                                value: new ItemModel({
                                    name,
                                    folderId: folder.id
                                })
                            });
                        }
                    }
                );
            }
        });
    },

    _closeAnalysis(evt) {
        evt.preventDefault();
        router.setQuery('analysis', null, { trigger: false });
        this.controlPanel.$el.addClass('hidden');
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
            ].join(','), { replace: true });
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
            var viewer = this.viewerWidget.viewer || {};
            if (viewer.zoomRange && annotation._pageElements === true) {
                annotation.setView(viewer.bounds(), viewer.zoom(), viewer.zoomRange().max, true);
            }
            annotation.set('loading', true);
            annotation.once('g:fetched', () => {
                annotation.unset('loading');
            });
            annotation.fetch().then(() => {
                // abandon this if the annotation should not longer be shown
                // or we are now showing a different image.
                if (!annotation.get('displayed') || annotation.get('itemId') !== this.model.id) {
                    return null;
                }
                this.viewerWidget.drawAnnotation(annotation);
                return null;
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

    _highlightAnnotationForInteractiveMode(annotation, element) {
        if (!this.annotationSelector.interactiveMode()) {
            return;
        }
        this._closeContextMenu();
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
        if (!this.viewerWidget) {
            return;
        }

        this.viewerWidget.removeAnnotation(
            new AnnotationModel({ _id: 'region-selection' })
        );
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
        this.viewerWidget.drawAnnotation(annotation, { fetch: false });
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
        if (annotationId === 'region-selection' || annotationId === 'selected' || !this.annotationSelector.interactiveMode()) {
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
        if (annotationId === 'region-selection' || annotationId === 'selected' || !this.annotationSelector.interactiveMode()) {
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
        if (annotationId === 'region-selection' || annotationId === 'selected') {
            return;
        }
        element.annotation = this.annotations.get(annotationId);
        if (element.annotation) {
            this.popover.collection.add(element);
        }
    },

    mouseOutAnnotation(element, annotationId) {
        if (annotationId === 'region-selection' || annotationId === 'selected') {
            return;
        }
        element.annotation = this.annotations.get(annotationId);
        if (element.annotation) {
            this.popover.collection.remove(element);
        }
    },

    mouseResetAnnotation() {
        if (this.popover.collection.length) {
            this.popover.collection.reset();
        }
    },

    mouseClickAnnotation(element, annotationId, evt) {
        if (!element.annotation) {
            // This is an instance of "selectedElements" and should be ignored.
            return;
        }

        /*
         * Click events on geojs features are triggered once per feature in a single animation frame.
         * Here we collect all click events occurring in a single animation frame and defer processing.
         * On the next frame, the queue is processed and the action is only performed on the "closest"
         * feature.  Here "closest" is determined by a fast heuristic--the one with a vertex closest
         * to the point clicked.  We can improve this heuristic as necessary.
         */
        this._queueMouseClickAction(element, annotationId, evt.data.geometry, evt.mouse.geo);
        if (this._mouseClickQueue.length > 1) {
            return;
        }

        window.requestAnimationFrame(() => {
            const { element, annotationId } = this._processMouseClickQueue();
            if (evt.mouse.buttonsDown.right) {
                this._openContextMenu(element.annotation.elements().get(element.id), annotationId, evt);
            } else if (evt.mouse.modifiers.ctrl) {
                this._toggleSelectElement(element.annotation.elements().get(element.id));
            }
        });
    },

    toggleLabels(options) {
        this.popover.toggle(options.show);
    },

    _queueMouseClickAction(element, annotationId, geometry, center) {
        let minimumDistance = Number.POSITIVE_INFINITY;
        if (geometry.type !== 'Polygon') {
            // We don't current try to resolve any other geometry type, for the moment,
            // any point or line clicked on will always be chosen over a polygon.
            minimumDistance = 0;
        } else {
            const points = geometry.coordinates[0];
            // use an explicit loop for speed
            for (let index = 0; index < points.length; index += 1) {
                const point = points[index];
                const dx = point[0] - center.x;
                const dy = point[1] - center.y;
                const distance = dx * dx + dy * dy;
                minimumDistance = Math.min(minimumDistance, distance);
            }
        }
        this._mouseClickQueue.push({ element, annotationId, value: minimumDistance });
    },

    _processMouseClickQueue(evt) {
        const sorted = _.sortBy(this._mouseClickQueue, _.property('value'));
        this._mouseClickQueue = [];
        return sorted[0];
    },

    _toggleInteractiveMode(interactive) {
        if (!interactive) {
            this.viewerWidget.highlightAnnotation();
            this.annotations.each((annotation) => {
                annotation.unset('highlight');
                if (this.drawWidget) {
                    annotation.elements().each((element) => {
                        this.drawWidget.trigger('h:mouseoff', element);
                    });
                }
            });
        }
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
    },

    _setAnnotationOpacity(opacity) {
        this.viewerWidget.setGlobalAnnotationOpacity(opacity);
    },

    _setAnnotationFillOpacity(opacity) {
        this.viewerWidget.setGlobalAnnotationFillOpacity(opacity);
    },

    _onKeyDown(evt) {
        if (evt.key === 'a') {
            this._showOrHideAnnotations();
        } else if (evt.key === 's') {
            this.annotationSelector.selectAnnotationByRegion();
        }
    },

    _trackMousePosition(evt) {
        this._currentMousePosition = {
            page: {
                x: evt.pageX,
                y: evt.pageY
            },
            client: {
                x: evt.clientX,
                y: evt.clientY
            }
        };
    },

    _showOrHideAnnotations() {
        if (this.annotations.any((a) => a.get('displayed'))) {
            this.annotationSelector.hideAllAnnotations();
        } else {
            this.annotationSelector.showAllAnnotations();
        }
    },

    _selectElementsByRegion() {
        this._selectElementsByRegionCanceled = false;
        this.viewerWidget.drawRegion().then((coord) => {
            if (this._selectElementsByRegionCanceled) {
                return this;
            }
            const boundingBox = {
                left: coord[0],
                top: coord[1],
                width: coord[2],
                height: coord[3]
            };
            this._resetSelection();
            const found = this.getElementsInBox(boundingBox);
            found.forEach(({ element }) => this._selectElement(element));
            if (this.selectedElements.length > 0 && this._currentMousePosition) {
                // fake an open context menu
                const { element, annotationId } = found[0];
                this._openContextMenu(element, annotationId, {
                    mouse: this._currentMousePosition
                });
            }
            this.trigger('h:selectedElementsByRegion', this.selectedElements);
            return this;
        });
    },

    _selectElementsByRegionCancel() {
        this.viewerWidget.annotationLayer.mode(null);
        this._selectElementsByRegionCanceled = true;
        this.trigger('h:selectedElementsByRegion', []);
    },

    getElementsInBox(boundingBox) {
        const lowerLeft = { x: boundingBox.left, y: boundingBox.top + boundingBox.height };
        const upperRight = { x: boundingBox.left + boundingBox.width, y: boundingBox.top };

        const results = [];
        this.viewerWidget.featureLayer.features().forEach((feature) => {
            const r = feature.boxSearch(lowerLeft, upperRight, { partial: false });
            r.found.forEach((feature) => {
                const annotationId = feature.properties ? feature.properties.annotation : null;
                const element = feature.properties ? feature.properties.element : null;
                if (element && element.id && annotationId) {
                    const annotation = this.annotations.get(annotationId);
                    results.push({
                        element: annotation.elements().get(element.id),
                        annotationId
                    });
                }
            });
        });
        return results;
    },

    _openContextMenu(element, annotationId, evt) {
        if (!this.selectedElements.get(element.id)) {
            this._resetSelection();
            this._selectElement(element);
        }

        if (!this.selectedElements.get(element.id)) {
            // If still not selected, then the user does not have access.
            return;
        }

        // Defer the context menu action into the next animation frame
        // to work around a problem with preventDefault on Windows
        window.setTimeout(() => {
            const $window = $(window);
            const menu = this.$('#h-annotation-context-menu');
            const position = evt.mouse.page;
            menu.removeClass('hidden');

            // adjust the vertical position of the context menu
            // == 0, above the bottom; < 0, number of pixels below the bottom
            // the menu height is bigger by 20 pixels due to extra padding
            const belowWindow = Math.min(0, $window.height() - position.y - menu.height() + 20);
            // ensure the top is not above the top of the window
            const top = Math.max(0, position.y + belowWindow);

            // Put the context menu to the left of the cursor if it is too close
            // to the right edge.
            const windowWidth = $window.width();
            const menuWidth = menu.width();
            let left = position.x;
            if (left + menuWidth > windowWidth) {
                left -= menuWidth;
            }
            left = Math.max(left, 0);

            menu.css({ left, top });
            if (this.popover.collection.length) {
                this.popover.collection.reset();
            }
            this._contextMenuActive = true;
        }, 1);
    },

    _closeContextMenu() {
        if (!this._contextMenuActive) {
            return;
        }
        this.$('#h-annotation-context-menu').addClass('hidden');
        this._resetSelection();
        if (this.popover.collection.length) {
            this.popover.collection.reset();
        }
        this._contextMenuActive = false;
    },

    _editElement(element) {
        const annotation = this.annotations.get(element.originalAnnotation);
        this._editAnnotation(annotation);
        editElement(annotation.elements().get(element.id));
    },

    _redrawSelection() {
        this.viewerWidget.removeAnnotation(this.selectedAnnotation);
        this.viewerWidget.drawAnnotation(this.selectedAnnotation, {fetch: false});
    },

    _selectElement(element) {
        // don't allow selecting annotations with no write access or
        // elements not associated with a real annotation.
        const annotation = (element.collection || {}).annotation;
        if (!annotation || annotation.get('_accessLevel') < AccessType.WRITE) {
            return;
        }

        var elementModel = this.selectedElements.add(element.attributes);
        elementModel.originalAnnotation = annotation;
        this.viewerWidget.highlightAnnotation(this.selectedAnnotation.id);
    },

    _unselectElement(element) {
        this.selectedElements.remove(element.id);
        if (!this.selectedElements.length) {
            this.viewerWidget.highlightAnnotation();
        }
    },

    _toggleSelectElement(element) {
        if (this.selectedElements.get(element.id)) {
            this._unselectElement(element);
        } else {
            this._selectElement(element);
        }
    },

    _resetSelection() {
        this.viewerWidget.highlightAnnotation();
        if (this.selectedElements.length) {
            this.selectedElements.reset();
        }
    },

    _saveSelection() {
        const groupedAnnotations = this.selectedElements.groupBy((element) => element.originalAnnotation.id);
        _.each(groupedAnnotations, (elements, annotationId) => {
            const annotation = this.annotations.get(annotationId);
            _.each(elements, (element) => { /* eslint-disable backbone/no-silent */
                const annotationElement = annotation.elements().get(element.id);
                // silence the event because we want to make one save call for each annotation.
                annotationElement.set(element.toJSON(), { silent: true });
                if (!element.get('group')) {
                    annotationElement.unset('group', { silent: true });
                }
            });
            if (!elements.length) {
                return;
            }
            const annotationData = _.extend({}, annotation.get('annotation'));
            annotationData.elements = annotation.elements().toJSON();
            annotation.set('annotation', annotationData);
        });
    },

    _removeSelection() {
        const groupedAnnotations = this.selectedElements.groupBy((element) => element.originalAnnotation.id);
        _.each(groupedAnnotations, (elements, annotationId) => { /* eslint-disable backbone/no-silent */
            // silence the event because we want to make one save call for each annotation.
            const elementsCollection = this.annotations.get(annotationId).elements();
            elementsCollection.remove(elements, { silent: true });
            elementsCollection.trigger('reset', elementsCollection);
        });
    }
});
export default ImageView;
