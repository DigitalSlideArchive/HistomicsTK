import events from 'girder/events';
import router from 'girder/router';

import { registerPluginNamespace } from 'girder/pluginUtils';
import { exposePluginConfig } from 'girder/utilities/PluginUtils';

// expose symbols under girder.plugins
import * as histomicstk from 'girder_plugins/HistomicsTK';

// import modules for side effects
import './views/itemList';
import './views/itemPage';

import ConfigView from './views/body/ConfigView';

const pluginName = 'HistomicsTK';
const configRoute = `plugins/${pluginName}/config`;

registerPluginNamespace(pluginName, histomicstk);

exposePluginConfig(pluginName, configRoute);

router.route(configRoute, 'HistomicsTKConfig', function () {
    events.trigger('g:navigateTo', ConfigView);
});
