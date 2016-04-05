histomicstk.views.Visualization = girder.View.extend({
    initialize: function () {
        // the map object
        this._map = null;

        // the rendered map layers
        this._layers = [];

        // control model for file widget
        this._controlModel = new histomicstk.models.Widget({
            type: 'file'
        });

        // control widget view
        this._controlView = new histomicstk.views.ControlWidget({
            parentView: this,
            model: this._controlModel
        });

        this.listenTo(this._controlModel, 'change', function (model) {
            var id = model.get('value');
            girder.restRequest({
                path: 'item/' + id
            })
            .then(_.bind(function (item) {
                item = new girder.models.ItemModel(item);
                return this.addItem(item);
            }, this))
            .fail(_.bind(function () {
                var info = {
                    text: 'Could not render item as an image',
                    type: 'danger',
                    timeout: 5000,
                    icon: 'attention'
                };
                girder.events.trigger('g:alert', info);
                this._controlView.invalid();
            }, this));
        });

        // fallback to canvas renderer rather than dom
        geo.gl.vglRenderer.fallback = function () {return 'canvas';};
    },

    /**
     * Create a map object with the given global bounds.
     * @TODO Either fix the map.maxBounds setter or recreate all the current layers
     * in the new map object.
     */
    _createMap: function (bounds) {
        if (this._map) {
            this._map.exit();
        }
        bounds.left = bounds.left || 0;
        bounds.top = bounds.top || 0;
        var w = bounds.right - bounds.left;
        var h = bounds.bottom - bounds.top;
        var interactor = geo.mapInteractor({
            zoomAnimation: false
        });

        this._map = geo.map({
            node: '<div style="width: 100%; height: 100%"/>',
            width: this.$el.width() || 100,
            height: this.$el.height() || 100,
            ingcs: '+proj=longlat +axis=esu',
            gcs: '+proj=longlat +axis=enu',
            maxBounds: bounds,
            clampBoundsX: false,
            clampBoundsY: false,
            center: {x: w / 2, y: h / 2},
            zoom: 0,
            discreteZoom: false,
            interactor: interactor
        });
        this.$('.h-visualization-body').empty().append(this._map.node());
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
                        maxLevel: tiles.levels,
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
        var layer = this._map.createLayer('feature', {renderer: 'vgl'});
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
            tileRounding: Math.ceil
        });

        // estimate the global bounds if not provided
        if (!opts.sizeX) {
            opts.sizeX = Math.pow(2, opts.maxLevel) * opts.tileWidth;
        }
        if (!opts.sizeY) {
            opts.sizeY = Math.pow(2, opts.maxLevel) * opts.tileHeight;
        }

        this._createMap({
            right: opts.sizeX - 1,
            bottom: opts.sizeY - 1
        });
        layer = this._map.createLayer('osm', opts);
        this._layers.push(layer);
        this._map.draw();
        return layer;
    },

    render: function () {

        this.$el.html(histomicstk.templates.visualization());
        this._controlView.setElement(this.$('.h-open-image-widget')).render();

        return this;
    },

    destroy: function () {
        this._map.exit();
        this.$el.empty();
        this._controlModel.destroy();
        girder.View.prototype.destroy.apply(this, arguments);
    }
});
