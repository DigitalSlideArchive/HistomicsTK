import Panel from 'girder_plugins/item_tasks/views/Panel';

import router from '../router';
import events from '../events';

import regionSelector from '../templates/panels/regionSelector.pug';
import '../stylesheets/panels/regionSelector.styl';

var RegionSelector = Panel.extend({
    events: {
        'click .h-select-region-button': 'selectRegion',
        'click .h-clear-region-button': 'clearRegion'
    },
    render() {
        this.$el.html(regionSelector({
            id: 'region-panel-container',
            title: 'Region'
        }));
    },
    selectRegion() {
        var bounds = router.getQuery('bounds').split(',');
        var left = parseFloat(bounds[0]);
        var top = parseFloat(bounds[1]);
        var right = parseFloat(bounds[2]);
        var bottom = parseFloat(bounds[3]);
        var obj = {left, right, top, bottom};
        this.$('.h-region-value').val(JSON.stringify(obj));
        events.trigger('h:select-region', obj);
    },
    clearRegion() {
        this.$('.h-region-value').val('');
        events.trigger('h:select-region', null);
    }
});

export default RegionSelector;
