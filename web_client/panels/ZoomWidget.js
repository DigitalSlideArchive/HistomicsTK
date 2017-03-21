import _ from 'underscore';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

import zoomWidget from '../templates/panels/zoomWidget.pug';
import '../stylesheets/panels/zoomWidget.styl';

var ZoomWidget = Panel.extend({
    events: {
        'click .h-zoom-button': '_zoomButton',
        'input .h-zoom-slider': '_zoomSliderInput',
        'change .h-zoom-slider': '_zoomSliderChange'
    },
    initialize() {
        this._maxMag = 20;
        this._maxZoom = 8;
        this._minZoom = 0;
        this._zoomChanged = _.bind(this._zoomChanged, this);
    },
    render() {
        var value = 0;
        if (this.viewer) {
            value = this.zoomToMagnification(this.viewer.zoom());
        }
        this.$el.html(zoomWidget({
            id: 'zoom-panel-container',
            title: 'Zoom',
            min: Math.log2(this.zoomToMagnification(this._minZoom)) - 0.01,
            max: Math.log2(this._maxMag) + 0.01,
            step: 0.01,
            value: Math.log2(value),
            enabled: !!this.viewer
        }));
        this._zoomSliderInput();
    },
    setViewer(viewer) {
        var geo = window.geo;
        var range;
        if (this.viewer) {
            this.viewer.geoOff(geo.event.zoom, this._zoomChanged);
        }
        this.viewer = viewer;
        if (this.viewer) {
            this.viewer.geoOn(geo.event.zoom, this._zoomChanged);
            range = this.viewer.zoomRange();
            this._maxZoom = range.max;
            this._minZoom = range.min;
        }
        return this;
    },
    setZoom(val) {
        this._setSliderValue(val);
        this._zoomSliderInput();
    },
    magnificationToZoom(magnification) {
        return this._maxZoom - Math.log2(20 / magnification);
    },
    zoomToMagnification(zoom) {
        return 20 * Math.pow(2, zoom - this._maxZoom);
    },
    _getSliderValue() {
        return Math.pow(2, parseFloat(this.$('.h-zoom-slider').val()));
    },
    _setSliderValue(val) {
        if (val > 0) {
            val = Math.log2(val);
        } else {
            val = 0;
        }
        this.$('.h-zoom-slider').val(val);
    },
    _zoomChanged() {
        if (!this.viewer) {
            return;
        }
        this.setZoom(this.zoomToMagnification(this.viewer.zoom()));
    },
    _zoomButton(evt) {
        this.setZoom(this.$(evt.target).data('value'));
        this._zoomSliderChange();
    },
    _zoomSliderInput() {
        var val = this._getSliderValue().toFixed(1);
        this.$('.h-zoom-value').text(val);
    },
    _zoomSliderChange() {
        if (this.viewer) {
            this.viewer.zoom(
                this.magnificationToZoom(this._getSliderValue())
            );
        }
    }
});

export default ZoomWidget;
