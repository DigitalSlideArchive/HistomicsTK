
import ControlWidget from 'girder_plugins/item_tasks/views/ControlWidget';
import { wrap } from 'girder/utilities/PluginUtils';

import events from '../../events';

import regionWidget from '../../templates/widget/regionWidget.pug';

/**
 * Append an event listener triggering drawing mode when the region
 * button is clicked.
 */
wrap(ControlWidget, 'initialize', function (initialize) {
    initialize.apply(this);
    this.events['click .h-select-region-button'] = function () {
        events.trigger('s:widgetDrawRegion', this.model);
    };
});

/**
 * Wrap the control widget to add a button for drawing a region
 * on the ImageView.
 */
wrap(ControlWidget, 'render', function (render) {
    render.call(this);
    if (this.model.get('type') === 'region') {
        const input = this.$('input');
        this.$('.g-control-item').append(regionWidget());
        this.$('.g-region-input-placeholder').replaceWith(input);
    }
    return this;
});
