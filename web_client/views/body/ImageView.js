import geo from 'geojs';
import { restRequest, apiRoot } from 'girder/rest';

import View from '../View';
import events from '../../events';

import imageTemplate from '../../templates/body/image.pug';
import '../../stylesheets/body/image.styl';

var ImageView = View.extend({
    initialize(settings) {
        this.listenTo(this.model, 'g:fetched', this._setModel);
        this.render();
        this.model.fetch();
    },
    render() {
        this.$el.html(imageTemplate());
    },
    destroy() {
        this._destroyed = true;
        if (this.image) {
            this.image.exit();
        }
        return View.prototype.destroy.apply(this, arguments);
    },
    _setModel() {
        this.largeImage = this.model.get('largeImage');
        if (!this.largeImage) {
            events.trigger('g:alert', {
                text: 'Invalid largeImage item',
                type: 'danger',
                timeout: 5000,
                icon: 'attention'
            });
        }
        restRequest({
            path: 'item/' + this.model.id + '/tiles'
        }).then((tiles) => this._createImage(tiles));
    },
    _createImage(tiles) {
        if (this._destroyed) {
            return;
        }
        if (this.image) {
            this.image.exit();
        }

        var w = tiles.sizeX, h = tiles.sizeY;
        var $el = this.$('.h-image-view-container');
        var mapW = $el.innerWidth(), mapH = $el.innerHeight();
        var mapParams = {
            node: $el,
            ingcs: '+proj=longlat +axis=esu',
            gcs: '+proj=longlat +axis=enu',
            maxBounds: {left: 0, top: 0, right: w, bottom: h},
            center: {x: w / 2, y: h / 2},
            min: Math.min(0, Math.floor(Math.log(Math.min(
                (mapW || tiles.tileWidth) / tiles.tileWidth,
                (mapH || tiles.tileHeight) / tiles.tileHeight)) / Math.log(2))),
            max: Math.ceil(Math.log(Math.max(
                w / tiles.tileWidth,
                h / tiles.tileHeight)) / Math.log(2)),
            clampBoundsX: true,
            clampBoundsY: true,
            zoom: 0
        };
        var maxLevel = mapParams.max;
        mapParams.unitsPerPixel = Math.pow(2, maxLevel);
        var layerParams = {
            useCredentials: true,
            url: apiRoot + '/item/' + this.model.id + '/tiles/zxy/{z}/{x}/{y}',
            maxLevel: mapParams.max,
            wrapX: false,
            wrapY: false,
            tileOffset: function () {
                return {x: 0, y: 0};
            },
            attribution: '',
            tileWidth: tiles.tileWidth,
            tileHeight: tiles.tileHeight,
            tileRounding: Math.ceil,
            tilesAtZoom: _.bind(function (level) {
                var scale = Math.pow(2, maxLevel - level);
                return {
                    x: Math.ceil(tiles.sizeX / tiles.tileWidth / scale),
                    y: Math.ceil(tiles.sizeY / tiles.tileHeight / scale)
                };
            }, this),
            tilesMaxBounds: _.bind(function (level) {
                var scale = Math.pow(2, maxLevel - level);
                return {
                    x: Math.floor(tiles.sizeX / scale),
                    y: Math.floor(tiles.sizeY / scale)
                };
            }, this)
        };
        this.viewer = geo.map(mapParams);
        this.viewer.createLayer('osm', layerParams);
    }
});

export default ImageView;
