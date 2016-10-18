/* global d3 */
histomicstk.views.Visualization = girder.View.extend({
    initialize: function () {
        // rendered annotation layers
        this._annotations = {};

        // the map object
        this._map = null;

        // the rendered map layers
        this._layers = [];

        // control model for file widget
        this._controlModel = histomicstk.dialogs.image.model;

        this._annotationList = new histomicstk.views.AnnotationSelectorWidget({
            parentView: this
        });

        // extra scaling factor to handle images that are not powers of 2 in size
        this._unitsPerPixel = 1;

        // prebind the onResize method so we can remove it on destroy
        this._onResize = _.bind(this._onResize, this);

        // create a debounced rerender method to delay rendering
        // until after user navigation is completed
        this._debouncedRender = _.debounce(_.bind(this._syncViewport, this), 300);

        // shared viewport object for annotation layers
        this.viewport = new girder.annotation.Viewport();
        this._onResize();
        $(window).resize(this._onResize);

        this.listenTo(this._controlModel, 'change:value', function (model) {
            if (!model.get('value')) {
                return;
            }

            var id = model.get('value').id;
            girder.restRequest({
                path: 'item/' + id
            })
            .then(_.bind(function (item) {
                this._controlModel.get('value').set(item);
                histomicstk.router.setQuery('image', id);
                return this.addItem(this._controlModel.get('value'));
            }, this))
            .fail(_.bind(function () {
                var info = {
                    text: 'Could not render item as an image',
                    type: 'danger',
                    timeout: 5000,
                    icon: 'attention'
                };
                girder.events.trigger('g:alert', info);
                histomicstk.router.setQuery('image', null, {replace: true});
            }, this));
        });

        this.listenTo(histomicstk.events, 'query:image', function (image) {
            var currentImage = this._controlModel.get('value') || {};
            if (image && currentImage.id !== image) {
                this._controlModel.set('value', new girder.models.ItemModel({_id: image}));
            } else if (!image) {
                this.removeItem();
                this._controlModel.set('value', null);
            }
        });

        // Set image bounds on URL query parameter change
        this.listenTo(histomicstk.events, 'query:bounds', this._boundsFromQuery);
    },

    /**
     * Set the displayed image bounds according to the given query string.
     */
    _boundsFromQuery: function (query) {
        if (!this._map || !query) {
            return;
        }
        var bounds = query.split(',').map(function (v) { return +v; });
        this._map.rotation(bounds[4] * Math.PI / 180);
        this._map.bounds({
            left: bounds[0],
            top: -bounds[1],
            right: bounds[2],
            bottom: -bounds[3]
        }, null);
    },

    /**
     * Create a map object with the given global bounds.
     */
    _createMap: function (bounds, tileWidth, tileHeight, sizeX, sizeY) {
        if (this._map) {
            // reset bounds query parameter on map exit
            histomicstk.router.setQuery('bounds', null, {replace: true, trigger: false});
            this._map.exit();
        }
        bounds.left = bounds.left || 0;
        bounds.top = bounds.top || 0;
        tileWidth = tileWidth || (bounds.right - bounds.left);
        tileHeight = tileHeight || (bounds.bottom - bounds.top);
        sizeX = sizeX || (bounds.right - bounds.left);
        sizeY = sizeY || (bounds.bottom - bounds.top);
        var interactor = geo.mapInteractor({
            zoomAnimation: false
        });
        var mapW = this.$el.width() || 100, mapH = this.$el.height() || 100;
        var minZoom = Math.min(0, Math.floor(Math.log(Math.min(
                (mapW || tileWidth) / tileWidth,
                (mapH || tileHeight) / tileHeight)) / Math.log(2))),
            maxZoom = Math.ceil(Math.log(Math.max(
                sizeX / tileWidth,
                sizeY / tileHeight)) / Math.log(2));

        var mapParams ={
            node: '<div style="width: 100%; height: 100%"/>',
            width: mapW,
            height: mapH,
            ingcs: '+proj=longlat +axis=esu',
            gcs: '+proj=longlat +axis=enu',
            maxBounds: bounds,
            min: minZoom,
            max: maxZoom,
            // to set min and max appropriately, we need to know the tile size
            // (using a single image may require the ability to zoom out).
            clampBoundsX: false,
            clampBoundsY: false,
            center: {x: (bounds.left + bounds.right) / 2,
                     y: (bounds.top + bounds.bottom) / 2},
            zoom: minZoom,
            discreteZoom: false,
            interactor: interactor,
            unitsPerPixel: Math.pow(2, maxZoom)
        };
        this._map = geo.map(mapParams);

        this._boundsFromQuery(histomicstk.router.getQuery('bounds'));
        this._syncViewport();
        this._map.geoOn(geo.event.pan, _.bind(this._onMouseNavigate, this));
        this.$('.h-visualization-body').empty().append(this._map.node());
    },

    renderAnnotationList: function (item) {
        if (this._annotationList.collection) {
            this.stopListening(this._annotationList);
        }
        this._annotationList
            .setItem(item)
            .setElement(this.$('.h-annotation-panel'))
            .render()
            .$el.removeClass('hidden');

        this.listenTo(this._annotationList.collection, 'change:displayed', this._toggleAnnotation);
    },

    /**
     * Add a map layer from a girder item object.  The girder item should contain a
     * single image file (the first encountered will be used) or be registered
     * as a tiled image via the large_image plugin.
     *
     * This method is async, errors encountered will trigger an error event and
     * reject the returned promise.  The promise is resolved with the added map
     * layer.
     *
     * @TODO: How do we pass in bounds?
     * @TODO: Allow multiple layers (don't reset map on add).
     */
    addItem: function (item) {
        var promise;

        this.resetAnnotations();
        this.renderAnnotationList(item);

        // first check if it is a tiled image
        if (item.id === 'test' || item.has('largeImage')) {
            promise = girder.restRequest({
                path: 'item/' + item.id + '/tiles'
            })
            .then(_.bind(function (tiles) {
                // It is a tile item
                return this.addTileLayer(
                    {
                        url: girder.apiRoot + '/item/' + item.id + '/tiles/zxy/{z}/{x}/{y}',
                        maxLevel: tiles.levels - 1,
                        tileWidth: tiles.tileWidth,
                        tileHeight: tiles.tileHeight,
                        sizeX: tiles.sizeX,
                        sizeY: tiles.sizeY
                    }
                );
            }, this));
        } else { // check for an image file
            promise = girder.restRequest({
                path: 'item/' + item.id + '/files',
                limit: 1 // we *could* look through all the files,
            })
            .then(_.bind(function (files) {
                var img = null, defer, imgElement;

                _.each(files, function (file) {
                    if ((file.mimeType || '').startsWith('image/')) {
                        img = new girder.models.FileModel(file);
                    }
                });

                if (!img) {
                    // no image is present, so return a reject promise
                    return new $.Deferred().reject('No renderable image file found').promise();
                }

                // Download the file to get the image size
                defer = new $.Deferred();
                imgElement = new Image();
                imgElement.onload = function () {
                    defer.resolve(imgElement);
                };
                imgElement.onerror = function () {
                    defer.reject('Could not load image');
                };
                imgElement.src = girder.apiRoot + '/file/' + img.id + '/download';
                return defer.promise();
            }))
            .then(_.bind(function (img) {
                // Inspect the image and add it to the map
                return this.addImageLayer(
                    img.src,
                    {
                        right: img.width,
                        bottom: img.height
                    }
                );
            }, this));
        }

        return promise;
    },

    /**
     * Add a single image as a quad feature on the viewer from an image url.
     *
     * @param {string} url The url of the image
     * @param {object} bounds
     *  The coordinate bounds of the image (left, right, top, bottom).
     */
    addImageLayer: function (url, bounds) {
        this._createMap(bounds);
        var layer = this._map.createLayer(
            'feature', {features: ['quad.imageCrop']});
        var quad = layer.createFeature('quad');
        quad.data([
            {
                ll: {x: bounds.left || 0, y: -bounds.bottom},
                ur: {x: bounds.right, y: bounds.top || 0},
                image: url
            }
        ]);
        this._layers.push(layer);
        this._map.draw();
        return quad;
    },

    removeItem: function () {
        this.resetAnnotations();
        this._annotationList.render().$el.addClass('hidden');
        if (this._map) {
            this._map.exit();
            this._map = null;
            this.$('.h-visualization-body').empty();
        }
    },

    /**
     * Add a tiled image as a tileLayer on the view from an options object.
     * The options object takes options supported by geojs's tileLayer:
     *
     *   https://github.com/OpenGeoscience/geojs/blob/master/src/tileLayer.js
     *
     * @param {object} opts Tile layer required options
     * @param {string|function} opts.url
     */
    addTileLayer: function (opts) {
        var layer;
        if (!_.has(opts, 'url')) {
            throw new Error('`url` parameter required.');
        }

        _.defaults(opts, {
            features: ['quad.imageCrop'],
            useCredentials: true,
            maxLevel: 10,
            wrapX: false,
            wrapY: false,
            tileOffset: function () {
                return {x: 0, y: 0};
            },
            attribution: '',
            tileWidth: 256,
            tileHeight: 256,
            tileRounding: Math.ceil,
            tilesAtZoom: function (level) {
                var scale = Math.pow(2, opts.maxLevel - level);
                return {
                    x: Math.ceil(opts.sizeX / opts.tileWidth / scale),
                    y: Math.ceil(opts.sizeY / opts.tileHeight / scale)
                };
            },
            tilesMaxBounds: function (level) {
                var scale = Math.pow(2, opts.maxLevel - level);
                return {
                    x: Math.floor(opts.sizeX / scale),
                    y: Math.floor(opts.sizeY / scale)
                };
            }
        });

        // estimate the global bounds if not provided
        if (!opts.sizeX) {
            opts.sizeX = Math.pow(2, opts.maxLevel) * opts.tileWidth;
        }
        if (!opts.sizeY) {
            opts.sizeY = Math.pow(2, opts.maxLevel) * opts.tileHeight;
        }

        this._createMap({
            right: opts.sizeX,
            bottom: opts.sizeY
        }, opts.tileWidth, opts.tileHeight, opts.sizeX, opts.sizeY);
        layer = this._map.createLayer('osm', opts);
        this._unitsPerPixel = this._map.unitsPerPixel(opts.maxLevel - 1);
        this._layers.push(layer);
        this._map.draw();
        return layer;
    },

    addAnnotationLayer: function (annotation) {
        var el = d3.select(this.el)
                .select('.h-annotation-layers')
                .append('svg')
                .node();

        this._onResize();
        var settings = $.extend({
            el: el,
            viewport: this.viewport
        }, annotation.annotation);

        this._annotations[annotation._id] = new girder.annotation.Annotation(settings).render();
        return this;
    },

    removeAnnotationLayer: function (id) {
        if (_.has(this._annotations, id)) {
            this._annotations[id].remove();
            delete this._annotations[id];
        }
        return this;
    },

    resetAnnotations: function () {
        _.each(_.keys(this._annotations), _.bind(this.removeAnnotationLayer, this));
        return this;
    },

    render: function () {

        this.$el.html(histomicstk.templates.visualization());
        return this;
    },

    destroy: function () {
        $(window).off('resize', this._onResize);
        this._map.exit();
        this.$el.empty();
        this._controlModel.destroy();
        girder.View.prototype.destroy.apply(this, arguments);
    },

    /**
     * Resize the viewport according to the size of the container.
     */
    _onResize: function () {
        var width = this.$el.width() || 100;
        var height = this.$el.height() || 100;
        var layers = this.$('.h-annotation-layers > svg');

        this.viewport.set({
            width: width,
            height: height
        });
        layers.attr('width', width);
        layers.attr('height', height);
        this._syncViewport();
    },

    /**
     * Respond to fast firing mouse navigation events by applying transforms
     * to the svg element so we can debounce the slow rerenders.
     */
    _onMouseNavigate: function () {
        var ul, dz;
        if (!this._lastCorner) {
            // do a full rerender if we haven't synced the viewport yet
            this._syncViewport();
            return;
        }

        ul = this._map.gcsToDisplay({
            x: this._lastCorner.x,
            y: this._lastCorner.y
        });

        dz = Math.pow(2, this._map.zoom() - this._lastZoom);

        this.$('.h-annotation-layers')
            .css(
                'transform',
                'translate(' + ul.x + 'px,' + ul.y + 'px)scale(' + dz + ')'
            );

        // schedule a debounced rerender
        this._debouncedRender();

        // Update the bounds in the query string
        var bounds = this._map.bounds(undefined, null);
        histomicstk.router.setQuery(
            'bounds',
            [
                this._formatNumber(bounds.left),
                this._formatNumber(-bounds.top),
                this._formatNumber(bounds.right),
                this._formatNumber(-bounds.bottom),
                this._formatNumber(this._map.rotation() * 180 / Math.PI)
            ].join(','),
            {
                replace: true,
                trigger: false
            }
        );
    },

    _formatNumber: function (num) {
        return (Math.round(num * 100) / 100).toString()
    },

    _syncViewport: function () {
        var bds;
        if (!this._map) {
            return;
        }

        this.$('.h-annotation-layers')
            .css('transform', '');

        bds = this._map.bounds(undefined, null);
        this._lastCorner = this._map.displayToGcs({
            x: 0,
            y: 0
        });
        this._lastZoom = this._map.zoom();

        this.viewport.set({
            scale: (bds.right - bds.left) / (this.viewport.get('width') * this._unitsPerPixel)
        });
        this.viewport.set({
            top: -bds.top / this._unitsPerPixel,
            left: bds.left / this._unitsPerPixel
        });
    },

    _toggleAnnotation: function (model) {
        if (model.get('displayed') && !_.has(this._annotations, model.id)) {
            girder.restRequest({
                path: 'annotation/' + model.id
            }).then(_.bind(this.addAnnotationLayer, this));
        } else if (!model.get('displayed') && _.has(this._annotations, model.id)) {
            this.removeAnnotationLayer(model.id);
        }
    }
});

histomicstk.dialogs.image = new histomicstk.views.ItemSelectorWidget({
    parentView: null,
    model: new histomicstk.models.Widget({
        type: 'file'
    })
});
