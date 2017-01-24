import GeojsViewer from 'girder_plugins/large_image/views/imageViewerWidget/geojs';
import SlicerPanelGroup from 'girder_plugins/slicer_cli_web/views/PanelGroup';

import events from '../../events';
import View from '../View';
// import ControlPanel from '../panels/ControlPanel';

import imageTemplate from '../../templates/body/image.pug';
import '../../stylesheets/body/image.styl';

var ImageView = View.extend({
    events: {},
    initialize(settings) {
        this.viewerWidget = null;
        events.trigger('h:imageOpened', null);
        this.controlPanel = new SlicerPanelGroup({
            parentView: this
        });
        this.render();
    },
    render() {
        this.$el.html(imageTemplate());
        if (this.model) {
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
        this.controlPanel.setElement('#h-control-panel-container').render();
    },
    destroy() {
        if (this.viewerWidget) {
            this.viewerWidget.destroy();
        }
        this.viewerWidget = null;
        events.trigger('h:imageOpened', null);
        return View.prototype.destroy.apply(this, arguments);
    }
});

export default ImageView;
