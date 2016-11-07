import * as histomicstk from '.';
import './routes';
import { registerPluginNamespace } from 'girder/pluginUtils';

registerPluginNamespace('HistomicsTK', histomicstk);

export default histomicstk;
