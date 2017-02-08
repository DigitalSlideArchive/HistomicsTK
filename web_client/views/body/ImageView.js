import _ from 'underscore';

import ItemModel from 'girder/models/ItemModel';
import GeojsViewer from 'girder_plugins/large_image/views/imageViewerWidget/geojs';
import SlicerPanelGroup from 'girder_plugins/slicer_cli_web/views/PanelGroup';

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
            });
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
     */
    _setImageInput() {
        if (!this.model.id) {
            return;
        }
        var image = this.model;
        _.each(this.controlPanel.models(), (model) => {
            if (model.get('type') === 'image') {
                model.set('value', image, {trigger: true});
            }
        });
    }
});

export default ImageView;
