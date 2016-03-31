histomicstk.views.Visualization = girder.View.extend({
    initialize: function () {
        // the map object
        this._map = null;

        // the rendered map layers
        this._layers = [];
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
        var w = bounds.right - bounds.left || 0;
        var h = bounds.bottom - bounds.top || 0;
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
            clampBoundsX: true,
            clampBoundsY: true,
            center: {x: w / 2, y: h / 2},
            zoom: 0,
            discreteZoom: false,
            interactor: interactor
        });
        this.$el.empty();
        this.$el.append(this._map.node());
    },

    /**
     * This will expand the attached map's maximum bounds to cover
     * the given bounds. {left, right, top, bottom} in pixel coordinates.
     */
    _expandBounds: function (bounds) {
        var mapBounds, newBounds = {
            left: bounds.left || 0,
            right: bounds.right,
            top: bounds.top || 0,
            bottom: bounds.bottom
        };
        if (this._map) {
            mapBounds = this._map.maxBounds();
            newBounds = {
                left: Math.min(bounds.left || 0, mapBounds.left),
                top: Math.min(bounds.top || 0, mapBounds.top),
                right: Math.max(bounds.right, mapBounds.right),
                bottom: Math.max(bounds.bottom, mapBounds.bottom)
            };
        }
        this._createMap(newBounds);
        return newBounds;
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
     */
    addItem: function (item) {
        var promise;

        // first check if it is a tiled image
        if (item.has('largeImage')) {
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
        this._expandBounds(bounds);
        var layer = this._map.createLayer('feature', {renderer: 'vgl'});
        var quad = layer.createFeature('quad');
        quad.data([
            {
                ll: {x: bounds.left || 0, y: bounds.bottom},
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

        this._expandBounds({
            right: opts.sizeX - 1,
            bottom: opts.sizeY - 1
        });
        layer = this._map.createLayer('osm', opts);
        this._layers.push(layer);
        this._map.draw();
        return layer;
    },

    render: function () {

        new girder.models.ItemModel({'_id': '56f55c0f62a8f80b77e45c68'})
            .on('change', _.bind(function (item) {
                this.addItem(item);
            }, this)).fetch();
        /*
        new girder.models.ItemModel({'_id': '56fd6d8c62a8f8692876ad89'})
            .on('change', _.bind(function (item) {
                this.addItem(item);
            }, this)).fetch();
        */

        return this;
    },

    destroy: function () {
        this._map.exit();
        this.$el.empty();
        girder.View.prototype.destroy.apply(this, arguments);
    }
});
