import _ from 'underscore';
import Panel from 'girder_plugins/item_tasks/views/Panel';

import zoomWidget from '../templates/panels/zoomWidget.pug';
import '../stylesheets/panels/zoomWidget.styl';

/**
 * Define a widget for controlling the view magnification with
 * a dynamic slider and buttons.
 *
 * Note: This code juggles three different representations
 * of the magnification:
 *
 *   * The "osm-like" zoom level:
 *       0: 1x1 global tiles
 *       1: 2x2 global tiles
 *       N: N+1xN+1 global tiles
 *
 *   * The image magnification:
 *       m = M 2^(z - Z)
 *         for max magnification M
 *             max zoom level Z
 *             current zoom level z
 *
 *   * The value of the range slider for log scaling:
 *       val = log2(m)
 *         for magnification m
 */
var ZoomWidget = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-zoom-button': '_zoomButton',
        'input .h-zoom-slider': '_zoomSliderInput',
        'change .h-zoom-slider': '_zoomSliderChange'
    }),
    initialize() {
        // set defaults that will be overwritten when a viewer is added
        this._maxMag = 20;
        this._maxZoom = 8;
        this._minZoom = 0;

        // bind the context of the viewer zoom handler
        this._zoomChanged = _.bind(this._zoomChanged, this);
    },
    render() {
        var value = 0;
        var min;
        var max;
        var buttons;

        if (this.viewer) {
            // get current magnification from the renderer
            value = this.zoomToMagnification(this.renderer.zoom());
        }

        // get the minimum value of the slider on a logarithmic scale
        // (here we expand the range slightly to make sure valid ranges
        // aren't clipped due to the slider step size)
        min = Math.log2(this.zoomToMagnification(this._minZoom)) - 0.01;
        max = Math.log2(this._maxMag) + 0.01;

        // get a list of discrete values to show as buttons
        buttons = _.filter([1, 2.5, 5, 10, 20, 40], (v) => v <= this._maxMag);
        buttons = _.last(buttons, 5);
        buttons = [0].concat(buttons);

        this.$el.html(zoomWidget({
            id: 'zoom-panel-container',
            title: 'Zoom',
            min: min,
            max: max,
            step: 0.01,
            value: Math.log2(value),
            disabled: !this.renderer,
            buttons: buttons
        }));

        // make the panel collapsible
        this.$('.g-panel-content').collapse({toggle: false});

        // set the text value on the readonly input box
        this._zoomSliderInput();
    },

    /**
     * Set the viewer instance and set several internal variables used
     * to convert between magnification and zoom level.
     */
    setViewer(viewer) {
        var geo = window.geo;
        var range;
        this.viewer = viewer;
        this.renderer = viewer.viewer;
        if (this.renderer) {
            this.renderer.geoOn(geo.event.zoom, this._zoomChanged);
            range = this.renderer.zoomRange();
            this._maxZoom = range.max;
            this._minZoom = range.min;
        }
        return this;
    },

    /**
     * Set the native magnification from the current image.  This
     * is given in the /item/{id}/tiles endpoint from large_image.
     */
    setMaxMagnification(magnification) {
        this._maxMag = magnification;
    },

    /**
     * Set the controls to the given magnification level.
     */
    setMagnification(val) {
        this._setSliderValue(val);
        this._zoomSliderInput();
    },

    /**
     * Convert from magnification to zoom level.
     */
    magnificationToZoom(magnification) {
        return this._maxZoom - Math.log2(this._maxMag / magnification);
    },

    /**
     * Convert from zoom level to magnification.
     */
    zoomToMagnification(zoom) {
        return this._maxMag * Math.pow(2, zoom - this._maxZoom);
    },

    /**
     * Get the value of the slider in magnification scale.
     */
    _getSliderValue() {
        return Math.pow(2, parseFloat(this.$('.h-zoom-slider').val()));
    },

    /**
     * Set the slider value to a specific magnification.
     */
    _setSliderValue(val) {
        if (val > 0) {
            val = Math.log2(val);
        } else {
            val = 0;
        }
        this.$('.h-zoom-slider').val(val);
    },

    /**
     * A handler called when the viewer's zoom level changes.
     */
    _zoomChanged() {
        if (!this.renderer) {
            return;
        }
        this.setMagnification(this.zoomToMagnification(this.renderer.zoom()));
    },

    /**
     * A handler called when one of the magnification buttons is clicked.
     */
    _zoomButton(evt) {
        this.setMagnification(this.$(evt.currentTarget).data('value'));
        this._zoomSliderChange();
    },

    /**
     * A handler called as the slider is moved.
     */
    _zoomSliderInput() {
        var val = this._getSliderValue().toFixed(1);
        this.$('.h-zoom-value').text(val);
    },

    /**
     * A handler that is called *after* the slider is moved.
     */
    _zoomSliderChange() {
        if (this.renderer) {
            this.renderer.zoom(
                this.magnificationToZoom(this._getSliderValue())
            );
        }
    }
});

export default ZoomWidget;
